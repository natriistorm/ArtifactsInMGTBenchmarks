from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 0
N_PER_CLASS = 500         
SENT_END = re.compile(r"([.!?])(\s+)")
_WS = re.compile(r"\s+")

DETECTORS = {
    "chatgpt-detector-roberta": "Hello-SimpleAI/chatgpt-detector-roberta",
    "openai-detector":          "openai-community/roberta-base-openai-detector",
    "radar":                    "TrustSafeAI/RADAR-Vicuna-7B",
    "roberta-mixed":            "andreas122001/roberta-mixed-detector",
    "fakespot":                 "fakespot-ai/roberta-base-ai-text-detection-v1",
    "piratexx":                 "PirateXX/AI-Content-Detector",
}


# --------------------------------------------------------------------------- #
# semantics guard                                                              #
# --------------------------------------------------------------------------- #

def semantic_key(t: str) -> str:
    """Everything the operators are allowed to change, removed. If two texts share
    this key they differ only in whitespace and punctuation *style*, never in
    words. NFKC first so unicode folding does not register as a word change."""
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", t).lower())


# --------------------------------------------------------------------------- #
# formatting profile                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class FormatProfile:
    """Measured formatting style of one class of one subset."""
    p_newline: float           # fraction of docs containing a newline
    line_lens: list[int]       # empirical distribution of line lengths (chars)
    p_double_space: float      # per sentence boundary, rate of >=2 spaces
    p_curly_quote: float       # of quote chars, fraction that are curly
    p_em_dash: float           # of dash chars, fraction that are en/em dashes
    mean_ws_runs: float        # whitespace runs per 1k chars (diagnostic only)

    def to_json(self) -> dict:
        d = asdict(self)
        d["line_lens"] = d["line_lens"][:2000]  # keep the artifact file small
        return d


def fit_profile(texts: list[str], rng: np.random.Generator) -> FormatProfile:
    n = len(texts) or 1
    with_nl = [t for t in texts if "\n" in t]
    line_lens: list[int] = []
    for t in with_nl:
        line_lens += [len(l) for l in t.split("\n") if l.strip()]
    if not line_lens:
        line_lens = [10_000]  # no wrapping observed -> effectively never wrap

    bnd = dsp = 0
    for t in texts:
        for m in SENT_END.finditer(t):
            bnd += 1
            if len(m.group(2)) >= 2 or "\n" in m.group(2):
                dsp += 1
    straight = sum(t.count('"') + t.count("'") for t in texts)
    curly = sum(len(re.findall(r"[‘’“”]", t)) for t in texts)
    hyph = sum(t.count("-") for t in texts)
    fancy = sum(len(re.findall(r"[–—]", t)) for t in texts)
    chars = sum(len(t) for t in texts) or 1

    return FormatProfile(
        p_newline=sum("\n" in t for t in texts) / n,
        line_lens=rng.permutation(np.array(line_lens))[:5000].tolist(),
        p_double_space=dsp / bnd if bnd else 0.0,
        p_curly_quote=curly / (curly + straight) if (curly + straight) else 0.0,
        p_em_dash=fancy / (fancy + hyph) if (fancy + hyph) else 0.0,
        mean_ws_runs=1000 * sum(len(re.findall(r"[ \t]{2,}", t)) for t in texts) / chars,
    )


# --------------------------------------------------------------------------- #
# operators                                                                    #
# --------------------------------------------------------------------------- #

def canon(text: str) -> str:
    """Remove extraction-inherited formatting. Words untouched."""
    t = unicodedata.normalize("NFKC", text)
    for a, b in [("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-")]:
        t = t.replace(a, b)
    return _WS.sub(" ", t).strip()


def inject(text: str, prof: FormatProfile, rng: np.random.Generator) -> str:
    """Re-impose `prof`'s formatting on `text`. Starts from canon() so the result
    reflects only `prof`, not the document's original formatting."""
    t = canon(text)

    # 1. punctuation style
    if prof.p_curly_quote > 0:
        t = "".join(
            ("“" if rng.random() < prof.p_curly_quote else c) if c == '"' else
            ("’" if rng.random() < prof.p_curly_quote else c) if c == "'" else c
            for c in t
        )
    if prof.p_em_dash > 0:
        t = re.sub(r"-", lambda _: "—" if rng.random() < prof.p_em_dash else "-", t)

    # 2. double spaces after sentence boundaries
    if prof.p_double_space > 0:
        t = SENT_END.sub(
            lambda m: m.group(1) + ("  " if rng.random() < prof.p_double_space else " "), t
        )

    # 3. line wrapping at the profile's line lengths
    if rng.random() < prof.p_newline and prof.line_lens:
        lens = np.asarray(prof.line_lens)
        out, cur, target = [], [], int(rng.choice(lens))
        for w in t.split(" "):
            cur.append(w)
            if sum(len(x) + 1 for x in cur) >= target:
                out.append(" ".join(cur))
                cur, target = [], int(rng.choice(lens))
        if cur:
            out.append(" ".join(cur))
        t = "\n".join(out)
    return t


# --------------------------------------------------------------------------- #
# build the counterfactual evaluation sets                                     #
# --------------------------------------------------------------------------- #

def build(data: Path, out: Path, n_per_class: int) -> None:
    files = [
        f for f in sorted(data.glob("*.parquet"))
        if "sample3000" not in f.name and "scientific_all" not in f.name
    ]
    out.mkdir(parents=True, exist_ok=True)
    profiles, rows, diag = {}, [], []

    for fp in files:
        name = fp.stem
        rng = np.random.default_rng(SEED)
        df = pd.read_parquet(fp, columns=["text", "label"])
        h_all = df.loc[df.label == 0, "text"].tolist()
        m_all = df.loc[df.label == 1, "text"].tolist()
        pH, pM = fit_profile(h_all, rng), fit_profile(m_all, rng)
        profiles[name] = {"human": pH.to_json(), "machine": pM.to_json()}

        k = min(n_per_class, len(h_all), len(m_all))
        h = list(rng.choice(np.array(h_all, dtype=object), k, replace=False))
        m = list(rng.choice(np.array(m_all, dtype=object), k, replace=False))

        for label, texts in [(0, h), (1, m)]:
            for i, t in enumerate(texts):
                variants = {
                    "original": t,
                    "canon": canon(t),
                    # the counterfactual: machine text dressed as human
                    "inject_human": inject(t, pH, rng),
                    # the placebo: machine text dressed as machine
                    "inject_machine": inject(t, pM, rng),
                }
                key = semantic_key(t)
                for v, vt in variants.items():
                    assert semantic_key(vt) == key, f"{name}/{label}/{i}/{v} changed words"
                    rows.append({"subset": name, "label": label, "doc_id": i,
                                 "variant": v, "text": vt})

        diag.append({
            "subset": name,
            "human_p_newline": round(pH.p_newline, 3), "machine_p_newline": round(pM.p_newline, 3),
            "human_p_double_space": round(pH.p_double_space, 3),
            "machine_p_double_space": round(pM.p_double_space, 3),
            "human_ws_runs_per1k": round(pH.mean_ws_runs, 2),
            "machine_ws_runs_per1k": round(pM.mean_ws_runs, 2),
            "human_median_line_len": int(np.median(pH.line_lens)),
            "machine_median_line_len": int(np.median(pM.line_lens)),
        })
        print(f"[built] {name:24s} k={k}  human p(nl)={pH.p_newline:.2f} "
              f"p(dsp)={pH.p_double_space:.2f} | machine p(nl)={pM.p_newline:.2f} "
              f"p(dsp)={pM.p_double_space:.2f}", flush=True)

    pd.DataFrame(rows).to_parquet(out / "counterfactual_sets.parquet", index=False)
    (out / "profiles.json").write_text(json.dumps(profiles, indent=1))
    D = pd.DataFrame(diag)
    D.to_csv(out / "format_profiles.csv", index=False)
    print(f"\nsemantics assertion passed on all {len(rows)} variants")
    print(D.to_string(index=False))
    print(f"\nwrote {out}/counterfactual_sets.parquet, profiles.json, format_profiles.csv")


# --------------------------------------------------------------------------- #
# scoring + ARS                                                                #
# --------------------------------------------------------------------------- #

def score(out: Path, detector: str, batch_size: int = 16, max_len: int = 512,
          device: str | None = None, fp16: bool = True) -> None:
    """Run a HF sequence classifier over every variant and compute ARS."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = DETECTORS.get(detector, detector)
    if device:
        dev = device
    else:
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "cpu":
        print("[warn] running on CPU. roberta-large over 40k documents will take hours; "
              "check that torch sees your accelerator (torch.cuda.is_available()).", flush=True)
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(repo).to(dev).eval()
    # fp16 halves the memory and roughly doubles throughput on CUDA; inference only, and
    # we threshold at 0.5 so the reduced precision cannot change a decision meaningfully.
    if dev == "cuda" and fp16:
        mdl = mdl.half()

    df = pd.read_parquet(out / "counterfactual_sets.parquet")

    # ---- which logit index means "machine"? ---------------------------------
    # NEVER trust the label map alone. Several widely-used public detectors ship
    # config.id2label = None (RADAR, PirateXX, desklib), and at least one ships a
    # reversed map (openai-community/roberta-base-openai-detector is
    # {0: 'Fake', 1: 'Real'}). Guessing index 1 would silently mirror-image every
    # ARS value, which looks like a strong result rather than an error. So we read
    # the label map if present, then *verify it empirically* against the gold
    # labels of the unmodified documents and flip if it disagrees.
    id2l = ({int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
            if getattr(mdl.config, "id2label", None) else {})
    machine_idx = next(
        (i for i, v in id2l.items()
         if any(s in v for s in ("fake", "machine", "ai", "chatgpt", "generated"))),
        1,
    )

    def _scores(texts: list[str], idx: int) -> np.ndarray:
        """P(machine) for every text. Batches are formed over length-sorted input so
        each batch pads to its own longest member instead of to the global maximum;
        with these subsets (median ~190 tokens, max_len 512) that removes most of the
        padding compute. Results are scattered back to the original order."""
        n = len(texts)
        out = np.empty(n, dtype=np.float32)
        order = np.argsort([len(t) for t in texts])
        with torch.no_grad():
            for k in range(0, n, batch_size):
                sel = order[k:k + batch_size]
                b = tok([texts[j] for j in sel], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(dev)
                out[sel] = mdl(**b).logits.softmax(-1)[:, idx].float().cpu().numpy()
                if k % (batch_size * 100) == 0 and n > 5000:
                    print(f"  {k}/{n}", flush=True)
        return out

    cal = df[df.variant == "original"]
    cal_s = _scores(cal.text.tolist(), machine_idx)
    auc = roc_auc_score(cal.label.to_numpy(), cal_s)
    if auc < 0.5:
        machine_idx = 1 - machine_idx
        cal_s = 1.0 - cal_s
        auc = 1.0 - auc
        print(f"[{detector}] label map disagreed with the data -- FLIPPED orientation")
    acc = float(((cal_s >= 0.5).astype(int) == cal.label.to_numpy()).mean())
    rec_m = float((cal_s[cal.label == 1] >= 0.5).mean())
    rec_h = float((cal_s[cal.label == 0] < 0.5).mean())
    print(f"[{detector}] id2label={id2l or 'MISSING'} -> machine index {machine_idx}; "
          f"calibration AUC={auc:.3f} acc={acc:.3f} recall(M)={rec_m:.3f} "
          f"recall(H)={rec_h:.3f}; device {dev}")
    if rec_m < 0.5:
        print(f"[{detector}] WARNING: machine recall {rec_m:.2f} -- this detector is "
              f"near-chance on machine documents here, so its correct decisions were "
              f"weakly held and its ARS is not evidence of artifact reliance. Report the "
              f"margin shift instead, or exclude from the ARS table.")
    orient = {"detector": detector, "machine_idx": machine_idx, "cal_auc": auc,
              "cal_acc": acc, "recall_machine": rec_m, "recall_human": rec_h}
    (out / f"orientation_{detector.replace('/', '_')}.json").write_text(json.dumps(orient, indent=1))
    s_all = _scores(df.text.tolist(), machine_idx)
    df["score"] = s_all                      # P(machine), for the margin-shift metric
    df["pred"] = (s_all >= 0.5).astype(np.int64)

    piv = df.pivot_table(index=["subset", "label", "doc_id"], columns="variant",
                         values="pred", aggfunc="first")
    # P(machine) per variant, for the threshold-free margin metric. The flip rate needs a
    # decision and therefore a threshold; likelihood-based scorers have no well-calibrated
    # threshold on this data (see the AUC/F1 gap), so their flip rate would partly reflect
    # an arbitrary cut. The margin shift is comparable across detector families.
    sc = df.pivot_table(index=["subset", "label", "doc_id"], columns="variant",
                        values="score", aggfunc="first")
    res = []
    for subset, g in piv.groupby(level="subset"):
        gm = g.xs(1, level="label")   # machine docs
        gh = g.xs(0, level="label")   # human docs
        sm = sc.xs(subset, level="subset").xs(1, level="label")   # index: doc_id
        # correct-and-then-flipped, under the counterfactual and under the placebo
        base_m = gm[gm.original == 1]
        base_h = gh[gh.original == 0]
        ars_m = float((base_m.inject_human == 0).mean()) if len(base_m) else float("nan")
        plc_m = float((base_m.inject_machine == 0).mean()) if len(base_m) else float("nan")
        ars_h = float((base_h.canon == 1).mean()) if len(base_h) else float("nan")
        # base_m is indexed by (subset, doc_id); sm only by doc_id -- align on doc_id
        ids = base_m.index.get_level_values("doc_id") if len(base_m) else []
        smb = sm.loc[sm.index.intersection(ids)] if len(base_m) else sm.iloc[:0]
        d_cf = float((smb.original - smb.inject_human).mean()) if len(smb) else float("nan")
        d_pl = float((smb.original - smb.inject_machine).mean()) if len(smb) else float("nan")
        # UNCONDITIONED margin: over *all* machine documents, not only the correctly
        # classified ones. Conditioning on correctness selects a non-random subset -- for a
        # detector with low machine recall, plausibly the documents whose formatting already
        # looks most machine-like, which are then the ones an injected human profile moves
        # most. Measured on RADAR, the conditioned margin on the arXiv subsets is +0.096 and
        # the unconditioned margin is -0.004: the entire effect was selection. Always report
        # both; if they disagree, the unconditioned value is the defensible one.
        u_cf = float((sm.original - sm.inject_human).mean())
        u_pl = float((sm.original - sm.inject_machine).mean())
        res.append({
            "subset": subset, "detector": detector,
            "acc_original": float((g.original == g.index.get_level_values("label")).mean()),
            "n_machine_correct": len(base_m), "n_human_correct": len(base_h),
            "ARS_machine": round(ars_m, 3),
            "ARS_machine_placebo": round(plc_m, 3),
            "ARS_human": round(ars_h, 3),
            "ARS_net": round(ars_m - plc_m, 3),
            # threshold-free: mean drop in P(machine), counterfactual minus placebo
            "margin_cf": round(d_cf, 4),
            "margin_placebo": round(d_pl, 4),
            "margin_net": round(d_cf - d_pl, 4),
            "margin_net_uncond": round(u_cf - u_pl, 4),
            "n_machine_total": int(len(sm)),
            "ARS": round(np.nanmean([ars_m, ars_h]), 3),
        })
    R = pd.DataFrame(res)
    p = out / f"ars_{detector.replace('/', '_')}.csv"
    R.to_csv(p, index=False)
    df.to_parquet(out / f"preds_{detector.replace('/', '_')}.parquet", index=False)
    print("\n" + R.to_string(index=False))
    print(f"\nwrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scientific_subsets")
    ap.add_argument("--out", default="results/cf")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--detector", default="chatgpt-detector-roberta")
    ap.add_argument("--device", default=None,
                    help="cuda | mps | cpu. Default: cuda if available, else mps, else cpu.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="raise this on a real GPU; 64-128 is comfortable for roberta-large")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 on CUDA")
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--prefetch", action="store_true",
                    help="download every detector in DETECTORS into the HF cache. Run this on a "
                         "login node before submitting: compute nodes are usually offline.")
    a = ap.parse_args()
    out = Path(a.out)
    if a.prefetch:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        for name, repo in DETECTORS.items():
            try:
                AutoTokenizer.from_pretrained(repo)
                AutoModelForSequenceClassification.from_pretrained(repo)
                print(f"  cached {name:26s} {repo}")
            except Exception as e:
                print(f"  FAILED {name:26s} {repo}: {type(e).__name__}: {e}")
        return
    if a.build:
        build(Path(a.data), out, a.n_per_class)
    if a.score:
        score(out, a.detector, batch_size=a.batch_size,
              device=a.device, fp16=not a.fp32)
    if not (a.build or a.score):
        ap.error("pass --build and/or --score")


if __name__ == "__main__":
    main()
