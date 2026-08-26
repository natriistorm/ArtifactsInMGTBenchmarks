"""
A. Shallow-feature probe: can a logistic regression over ~25 content-free
   surface features separate human from machine text as well as a fine-tuned
   detector? Reports per-subset macro-F1 / AUC with bootstrap CIs, plus the
   single-feature AUC of every feature so the artifact is *named*, not just
   measured.

B. Format-normalization intervention: the causal test. Collapse whitespace
   runs, strip newlines, NFKC-fold unicode punctuation, then re-run the
   identical probe. A large drop means the benchmark's separability lived in
   document formatting inherited from extraction, not in language.

Usage
-----
    python shallow_probe.py                       # all full subsets
    python shallow_probe.py --sample3000          # the balanced 3k samples
    python shallow_probe.py --out results/probe   # writes .csv + .md

Outputs (under --out, default 'results/probe'):
    probe_main.csv        one row per (subset, variant): F1, AUC, CIs, TF-IDF ref
    probe_features.csv    one row per (subset, variant, feature): single-feature AUC
    probe_main.md         main-table fragment, ready to paste
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*encountered in.*", category=RuntimeWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")

SEED = 0
N_PER_CLASS = 10_000
N_BOOT = 1000
N_FOLDS = 5


_SENT_SPLIT = re.compile(r"[.!?]+")
_WS_RUN = re.compile(r"[ \t]{2,}")
_LATEX = re.compile(r"\\(cite|ref|label|begin|end|textit|textbf|emph)\b")
_CURLY_Q = re.compile(r"[\u2018\u2019\u201c\u201d]")
_FANCY_DASH = re.compile(r"[\u2013\u2014]")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")


def features(text: str) -> dict[str, float]:
    """~25 surface features. Keys are stable and used as table row labels."""
    n = len(text) + 1e-9
    words = text.split()
    nw = len(words) + 1e-9
    lens = [len(w) for w in words] or [0.0]
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    ns = len(sents) + 1e-9
    lower = [w.lower() for w in words]
    counts = pd.Series(lower).value_counts() if words else pd.Series(dtype=int)

    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sents": len(sents),
        "mean_sent_len": nw / ns,
        "mean_word_len": float(np.mean(lens)),
        "std_word_len": float(np.std(lens)),
        "ttr": len(set(lower)) / nw,
        "hapax_rate": float((counts == 1).sum()) / nw if words else 0.0,
        "r_comma": text.count(",") / n,
        "r_period": text.count(".") / n,
        "r_semicolon": text.count(";") / n,
        "r_colon": text.count(":") / n,
        "r_paren": text.count("(") / n,
        "r_bracket": text.count("[") / n,
        "r_ascii_quote": text.count('"') / n,
        "r_hyphen": text.count("-") / n,
        "r_exclam": text.count("!") / n,
        "r_question": text.count("?") / n,
        "r_newline": text.count("\n") / n,
        "r_ws_run": len(_WS_RUN.findall(text)) / n,
        "r_curly_quote": len(_CURLY_Q.findall(text)) / n,
        "r_fancy_dash": len(_FANCY_DASH.findall(text)) / n,
        "r_nonascii": sum(ord(c) > 127 for c in text) / n,
        "r_digit": sum(c.isdigit() for c in text) / n,
        "r_upper": sum(c.isupper() for c in text) / n,
        "r_number_tok": len(_NUM.findall(text)) / nw,
        "r_latex": len(_LATEX.findall(text)) / nw,
    }


FORMATTING_FEATURES = ["r_newline", "r_ws_run", "r_curly_quote", "r_fancy_dash"]


def normalize(text: str) -> str:
    """Exp. B intervention: remove formatting that extraction, not authorship,
    put in the document. NFKC folds curly quotes / fancy dashes / ligatures to
    ASCII; whitespace runs and newlines collapse to a single space."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC. ~50x faster than sklearn's in a bootstrap loop."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def _macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Binary macro-F1 from counts, avoiding sklearn's per-call overhead."""
    tp = np.count_nonzero((y == 1) & (pred == 1))
    fp = np.count_nonzero((y == 0) & (pred == 1))
    fn = np.count_nonzero((y == 1) & (pred == 0))
    tn = np.count_nonzero((y == 0) & (pred == 0))
    dp, dn = 2 * tp + fp + fn, 2 * tn + fn + fp
    return float(((2 * tp / dp if dp else 0.0) + (2 * tn / dn if dn else 0.0)) / 2)


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def bootstrap_ci(y, pred, proba, n_boot=N_BOOT, seed=SEED):
    """Percentile CIs for macro-F1 and AUC over the out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    f1s, aucs = [], []
    n = len(y)
    for _ in range(n_boot):
        b = rng.integers(0, n, size=n)
        yb = y[b]
        if yb.sum() in (0, n):
            continue
        f1s.append(_macro_f1(yb, pred[b]))
        aucs.append(_auc(yb, proba[b]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(f1s), q(aucs)


def eval_variant(texts: list[str], y: np.ndarray, tfidf: bool = True) -> tuple[dict, pd.DataFrame]:
    """Out-of-fold shallow probe + single-feature AUCs + optional TF-IDF reference."""
    F = pd.DataFrame([features(t) for t in texts])
    X = F.values
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    proba = cross_val_predict(_lr(), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    pred = (proba >= 0.5).astype(np.int64)
    (f1_lo, f1_hi), (auc_lo, auc_hi) = bootstrap_ci(y, pred, proba)

    row = {
        "shallow_f1": f1_score(y, pred, average="macro"),
        "shallow_f1_lo": f1_lo,
        "shallow_f1_hi": f1_hi,
        "shallow_auc": roc_auc_score(y, proba),
        "shallow_auc_lo": auc_lo,
        "shallow_auc_hi": auc_hi,
    }

    for name, cols in [
        ("lenonly", ["n_words"]),
        ("fmtonly", FORMATTING_FEATURES),
        ("nofmt", [c for c in F.columns if c not in FORMATTING_FEATURES]),
    ]:
        Xa = F[cols].values
        pa = cross_val_predict(_lr(), Xa, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        row[f"{name}_auc"] = roc_auc_score(y, pa)
        row[f"{name}_f1"] = f1_score(y, (pa >= 0.5).astype(int), average="macro")

    if tfidf:
        tf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=3000),
        )
        tp = cross_val_predict(tf, texts, y, cv=cv, n_jobs=-1)
        row["tfidf_f1"] = f1_score(y, tp, average="macro")

    feats = []
    for c in F.columns:
        v = F[c].values
        a = roc_auc_score(y, v) if np.ptp(v) > 0 else 0.5
        feats.append({"feature": c, "auc": max(a, 1 - a), "signed_auc": a})
    fdf = pd.DataFrame(feats).sort_values("auc", ascending=False)
    return row, fdf


def balanced_sample(df: pd.DataFrame, n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    k = min(n_per_class, df.label.value_counts().min())
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(k, random_state=SEED))
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scientific_subsets")
    ap.add_argument("--sample3000", action="store_true", help="use the 3k balanced samples")
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--no-tfidf", action="store_true", help="skip the TF-IDF reference (faster)")
    args = ap.parse_args()

    d = Path(args.data)
    pat = "*_sample3000.parquet" if args.sample3000 else "*.parquet"
    files = [
        f for f in sorted(d.glob(pat))
        if (("sample3000" in f.name) == args.sample3000) and "scientific_all" not in f.name
    ]
    if not files:
        raise FileNotFoundError(f"no subset parquets under {d}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    main_rows, feat_rows = [], []
    for fp in files:
        name = fp.stem.replace("_sample3000", "")
        df = balanced_sample(pd.read_parquet(fp, columns=["text", "label"]))
        y = df.label.to_numpy()
        print(f"\n=== {name}  n={len(df)} ({int((y == 0).sum())} human / {int((y == 1).sum())} machine) ===")

        tfidf_cache: float | None = None
        for variant, texts in [
            ("raw", df.text.tolist()),
            ("normalized", [normalize(t) for t in df.text]),
        ]:
            want_tfidf = (not args.no_tfidf) and variant == "raw"
            row, fdf = eval_variant(texts, y, tfidf=want_tfidf)
            if want_tfidf:
                tfidf_cache = row.get("tfidf_f1")
            elif tfidf_cache is not None:
                row["tfidf_f1"] = tfidf_cache
            row = {"subset": name, "variant": variant, "n": len(df), **row}
            main_rows.append(row)
            fdf.insert(0, "variant", variant)
            fdf.insert(0, "subset", name)
            feat_rows.append(fdf)
            top = ", ".join(f"{r.feature}={r.auc:.3f}" for r in fdf.head(3).itertuples())
            print(
                f"  {variant:11s} F1={row['shallow_f1']:.3f} "
                f"[{row['shallow_f1_lo']:.3f},{row['shallow_f1_hi']:.3f}] "
                f"AUC={row['shallow_auc']:.3f}  len-only={row['lenonly_auc']:.3f} "
                f"fmt-only={row['fmtonly_auc']:.3f} no-fmt={row['nofmt_f1']:.3f}"
                + (f" tfidf={row['tfidf_f1']:.3f}" if 'tfidf_f1' in row else "")
                + f"\n              top: {top}"
            )

        raw, nrm = main_rows[-2], main_rows[-1]
        print(f"  >> Δ(raw→norm) F1 = {raw['shallow_f1'] - nrm['shallow_f1']:+.3f}")

    M = pd.DataFrame(main_rows)
    F = pd.concat(feat_rows, ignore_index=True)
    M.to_csv(f"{out}_main.csv", index=False)
    F.to_csv(f"{out}_features.csv", index=False)

    piv = M.pivot(index="subset", columns="variant")
    lines = [
        "| Subset | Shallow-F1 (raw) | 95% CI | Shallow-F1 (norm) | Δ | len-only AUC | fmt-only AUC | TF-IDF F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in piv.index:
        r, nv = piv.loc[s, ("shallow_f1", "raw")], piv.loc[s, ("shallow_f1", "normalized")]
        tf = piv.loc[s, ("tfidf_f1", "raw")] if ("tfidf_f1", "raw") in piv.columns else float("nan")
        lines.append(
            f"| {s} | {r:.3f} | [{piv.loc[s, ('shallow_f1_lo','raw')]:.3f}, "
            f"{piv.loc[s, ('shallow_f1_hi','raw')]:.3f}] | {nv:.3f} | {r - nv:+.3f} | "
            f"{piv.loc[s, ('lenonly_auc','raw')]:.3f} | {piv.loc[s, ('fmtonly_auc','raw')]:.3f} | "
            f"{tf:.3f} |"
        )
    Path(f"{out}_main.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}_main.csv, {out}_features.csv, {out}_main.md")


if __name__ == "__main__":
    main()
