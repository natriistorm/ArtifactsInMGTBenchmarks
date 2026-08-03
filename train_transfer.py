#!/usr/bin/env python3
"""
train_transfer.py

Experiments E5-E7: the mDeBERTa side of the paper.

Trains a detector per (subset, condition) and evaluates every model on every
subset's held-out test set, producing decontaminated transfer matrices. The
conditions are the ACE ablation:

    random_raw        random split, raw text            <- the standard protocol
                                                           papers report
    cluster_raw       near-dup-cluster-aware split      <- isolates the
                                                           duplication leak (F2)
    cluster_canon     + format canonicalization         <- isolates the
                                                           format artifact (F1)
    cluster_canon_poe + artifact-debiased training      <- full ACE

Debiasing uses the shallow surface probe from shallow_probe.py as an explicit
bias-only model, fit on the training split only. Each example is reweighted by
(1 - p_bias(y_i|x_i))^gamma, so gradient mass moves to documents the artifact
cannot explain. At inference the detector is used alone.

Transfer cells are decontaminated: when evaluating a model trained on A against
subset B's test set, documents in B_test with a near-duplicate in A_train are
excluded, since F3 showed 40-80% overlap across the M4/SemEval24/COLING25 lineage.

Usage
-----
    python train_transfer.py --prepare                 # splits + clusters + bias probes
    python train_transfer.py --train --conditions random_raw cluster_canon_poe
    python train_transfer.py --train --conditions cluster_raw cluster_canon \
                             --subsets M4_arxiv SemEval24_arxiv IDMGSP_sci_paper
    python train_transfer.py --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contamination import (
    N_BANDS, MERSENNE, N_PERM, band_keys, build_index, matched_mask, norm_text, signatures, _h64,
)
from format_counterfactual import canon
from shallow_probe import features

SEED = 0                
MODEL = "microsoft/mdeberta-v3-base"
N_TRAIN = 4000          
N_TEST = 1000          
LR = 2e-5
BATCH = 8
EPOCHS = 1
GAMMA = 1.0
WEIGHT_CLIP = 10.0      # max weight, in units of the mean weight
LONG_SUBSETS = {"M4_peerread", "IDMGSP_sci_paper"}   # median >1k tokens -> 512
CONDITIONS = ["random_raw", "cluster_raw", "cluster_canon", "cluster_canon_poe"]
WORK = Path("results/transfer")


# --------------------------------------------------------------------------- #
# splits                                                                       #
# --------------------------------------------------------------------------- #

def cluster_ids(texts: list[str], threshold: float = 0.8) -> np.ndarray:
    """Union-find over LSH candidate pairs -> a near-duplicate cluster id per doc.
    Documents in the same cluster must never straddle the train/test boundary."""
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, MERSENNE, size=N_PERM, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=N_PERM, dtype=np.uint64)
    sig = signatures(texts, a, b)
    keys = band_keys(sig)
    idx = build_index(keys)

    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    # exact duplicates first (cheap and catches the bulk of the M4 redundancy)
    by_hash: dict[int, int] = {}
    for i, t in enumerate(texts):
        h = _h64(norm_text(t))
        if h in by_hash:
            union(by_hash[h], i)
        else:
            by_hash[h] = i

    for i in range(len(texts)):
        cand: set[int] = set()
        for bi in range(N_BANDS):
            cand.update(idx[bi].get(int(keys[bi][i]), ()))
        cand.discard(i)
        if not cand:
            continue
        c = np.fromiter(cand, dtype=np.int64, count=len(cand))
        for j in c[np.mean(sig[c] == sig[i], axis=1) >= threshold]:
            union(i, int(j))
    return np.array([find(i) for i in range(len(texts))])


def make_splits(df: pd.DataFrame, cid: np.ndarray, rng: np.random.Generator,
                cluster_aware: bool) -> tuple[np.ndarray, np.ndarray]:
    """Balanced train/test index arrays. If cluster_aware, whole clusters are
    assigned to one side so no near-duplicate spans the boundary."""
    y = df.label.to_numpy()
    if not cluster_aware:
        tr, te = [], []
        for lab in (0, 1):
            i = np.where(y == lab)[0]
            i = rng.permutation(i)
            te += list(i[: N_TEST // 2])
            tr += list(i[N_TEST // 2: N_TEST // 2 + N_TRAIN // 2])
        return np.array(tr), np.array(te)

    # assign clusters to test until the per-class quota is met, then to train
    order = rng.permutation(np.unique(cid))
    want_te = {0: N_TEST // 2, 1: N_TEST // 2}
    want_tr = {0: N_TRAIN // 2, 1: N_TRAIN // 2}
    tr, te = [], []
    got_te = {0: 0, 1: 0}
    got_tr = {0: 0, 1: 0}
    for c in order:
        mem = np.where(cid == c)[0]
        labs = y[mem]
        # a cluster goes wholly to test if it still fits the test quota
        if all(got_te[l] + int((labs == l).sum()) <= want_te[l] for l in (0, 1)) and \
           any(got_te[l] < want_te[l] for l in (0, 1)):
            te += list(mem)
            for l in (0, 1):
                got_te[l] += int((labs == l).sum())
        else:
            for l in (0, 1):
                take = [i for i in mem if y[i] == l][: max(0, want_tr[l] - got_tr[l])]
                tr += take
                got_tr[l] += len(take)
    return np.array(tr), np.array(te)


def prepare(data: Path) -> None:
    """Compute clusters, both split protocols, and the bias-model probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    WORK.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(data.glob("*.parquet"))
             if "sample3000" not in f.name and "scientific_all" not in f.name]

    for fp in files:
        name = fp.stem
        out = WORK / f"{name}.parquet"
        if out.exists():
            print(f"[skip] {name} already prepared")
            continue
        rng = np.random.default_rng(SEED)
        df = pd.read_parquet(fp, columns=["text", "label"])
        # cap before clustering: MinHash over 250k docs is not worth the time and
        # the splits only need N_TRAIN + N_TEST documents anyway
        cap = 25_000
        if len(df) > cap:
            df = df.sample(cap, random_state=SEED).reset_index(drop=True)
        print(f"[prep] {name} n={len(df)} clustering...", flush=True)
        cid = cluster_ids(df.text.tolist())
        n_clust = len(np.unique(cid))
        df["cluster"] = cid
        df["canon_text"] = [canon(t) for t in df.text]

        for tag, ca in [("random", False), ("cluster", True)]:
            tr, te = make_splits(df, cid, np.random.default_rng(SEED), ca)
            df[f"split_{tag}"] = "unused"
            df.loc[tr, f"split_{tag}"] = "train"
            df.loc[te, f"split_{tag}"] = "test"
            print(f"       {tag:8s} train={len(tr)} test={len(te)}")

        # bias-only model: fit on the cluster-split TRAIN only, predict everywhere
        tr_mask = (df.split_cluster == "train").to_numpy()
        F = pd.DataFrame([features(t) for t in df.text])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(F[tr_mask].values, df.label[tr_mask])
        p = clf.predict_proba(F.values)
        # probability the bias model assigns to the TRUE label
        df["p_bias"] = p[np.arange(len(df)), df.label.to_numpy()]
        df.to_parquet(out, index=False)
        print(f"       clusters={n_clust} ({100 * n_clust / len(df):.1f}% of docs)  "
              f"mean p_bias(train)={df.p_bias[tr_mask].mean():.3f}")


# --------------------------------------------------------------------------- #
# training                                                                     #
# --------------------------------------------------------------------------- #

def condition_spec(cond: str) -> tuple[str, str, bool]:
    """-> (split column tag, text column, use debiasing)"""
    return {
        "random_raw":        ("random",  "text",       False),
        "cluster_raw":       ("cluster", "text",       False),
        "cluster_canon":     ("cluster", "canon_text", False),
        "cluster_canon_poe": ("cluster", "canon_text", True),
    }[cond]


def train_one(name: str, cond: str, device: str, seed: int = 0,
              batch: int = BATCH, amp: bool = False, smoke: int = 0) -> Path:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    # The backbone must appear in the filename. Without it, switching MODEL silently
    # collides with the previous backbone's checkpoints: the [skip] branch fires and the
    # old results get read back as if they were the new model's.
    tag_model = MODEL.split("/")[-1]
    ckpt = WORK / f"model_{name}__{cond}__{tag_model}__s{seed}{'__smoke' if smoke else ''}.json"
    if ckpt.exists():
        print(f"[skip] {name}/{cond}/s{seed} already trained")
        return ckpt

    torch.manual_seed(seed)
    np.random.seed(seed)
    tag, textcol, debias = condition_spec(cond)
    df = pd.read_parquet(WORK / f"{name}.parquet")
    tr = df[df[f"split_{tag}"] == "train"].reset_index(drop=True)
    if smoke:
        tr = tr.groupby("label", group_keys=False).head(smoke // 2).reset_index(drop=True)
    max_len = 512 if name in LONG_SUBSETS else 256

    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

    w = np.ones(len(tr), dtype=np.float32)
    if debias:
        w = np.power(1.0 - tr.p_bias.to_numpy(dtype=np.float32), GAMMA)
        w = w / max(w.mean(), 1e-8)   # keep the effective learning rate comparable
        w = np.clip(w, 0.0, WEIGHT_CLIP)
        w = w / max(w.mean(), 1e-8)
        print(f"   weights: mean={w.mean():.3f} max={w.max():.2f} "
              f"frac<0.1={np.mean(w < 0.1):.2f}", flush=True)

    class DS(Dataset):
        def __len__(self): return len(tr)
        def __getitem__(self, i):
            return tr[textcol][i], int(tr.label[i]), float(w[i])

    def collate(batch):
        t, y, ww = zip(*batch)
        enc = tok(list(t), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return enc, torch.tensor(y), torch.tensor(ww)

    dl = DataLoader(DS(), batch_size=batch, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR)
    total = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    lossf = torch.nn.CrossEntropyLoss(reduction="none")
    # bf16 autocast on CUDA only; MPS autocast is not reliable for this model
    actx = (torch.autocast("cuda", dtype=torch.bfloat16) if (amp and device == "cuda")
            else torch.autocast("cpu", enabled=False))

    mdl.train()
    print(f"[train] {name}/{cond}/s{seed} n={len(tr)} steps={total} max_len={max_len} "
          f"batch={batch} debias={debias} device={device}", flush=True)
    for ep in range(EPOCHS):
        for step, (enc, y, ww) in enumerate(dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            with actx:
                out = mdl(**enc).logits
                loss = (lossf(out.float(), y.to(device)) * ww.to(device)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            if step % 100 == 0:
                print(f"   step {step}/{len(dl)} loss={loss.item():.4f}", flush=True)

    # evaluate on every subset's test set (matched text preprocessing)
    mdl.eval()
    rows = []
    for fp in sorted(WORK.glob("*.parquet")):
        tname = fp.stem
        if tname.startswith("model_"):
            continue
        tdf = pd.read_parquet(fp, columns=["text", "canon_text", "label", "split_cluster"])
        te = tdf[tdf.split_cluster == "test"].reset_index(drop=True)
        if smoke:
            te = te.groupby("label", group_keys=False).head(smoke // 2).reset_index(drop=True)
        texts, ys = te[textcol].tolist(), te.label.to_numpy()
        preds = np.empty(len(texts), dtype=np.int64)
        # length-sorted batching: pad to each batch's own maximum, not the global one
        order = np.argsort([len(t) for t in texts])
        eb = max(32, batch)
        with torch.no_grad():
            for k in range(0, len(texts), eb):
                sel = order[k:k + eb]
                enc = tok([texts[j] for j in sel], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to(device)
                preds[sel] = mdl(**enc).logits.argmax(-1).cpu().numpy()
        rows.append({"target": tname, "y": ys.tolist(), "pred": preds.tolist(),
                     "doc_hash": [_h64(norm_text(t)) for t in te.text]})
    ckpt.write_text(json.dumps(
        {"train": name, "condition": cond, "seed": seed, "model": MODEL, "evals": rows}))
    del mdl
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    print(f"[done] {name}/{cond}/s{seed} -> {ckpt}", flush=True)
    return ckpt


# --------------------------------------------------------------------------- #
# reporting                                                                    #
# --------------------------------------------------------------------------- #

def report() -> None:
    """Decontaminated transfer matrices + Delta_gen per condition."""
    from sklearn.metrics import f1_score

    # near-dup masks: which test docs of B appear in A's training split?
    subs = [f.stem for f in sorted(WORK.glob("*.parquet")) if not f.stem.startswith("model_")]
    train_hashes = {}
    for s in subs:
        df = pd.read_parquet(WORK / f"{s}.parquet")
        train_hashes[s] = set(
            _h64(norm_text(t)) for t in df[df.split_cluster == "train"].text
        )

    rows = []
    for ck in sorted(WORK.glob("model_*.json")):
        d = json.loads(ck.read_text())
        A, cond, seed = d["train"], d["condition"], d.get("seed", 0)
        for ev in d["evals"]:
            B = ev["target"]
            y = np.array(ev["y"]); p = np.array(ev["pred"])
            h = np.array(ev["doc_hash"], dtype=np.uint64)
            keep = np.array([hh not in train_hashes[A] for hh in h.tolist()]) if A != B else np.ones(len(y), bool)
            rows.append({
                "condition": cond, "seed": seed, "train": A, "target": B, "in_domain": A == B,
                "f1": f1_score(y, p, average="macro"),
                "f1_decontam": f1_score(y[keep], p[keep], average="macro") if keep.sum() > 20 else np.nan,
                "n_test": len(y), "n_kept": int(keep.sum()),
                "pct_contaminated": round(100 * (1 - keep.mean()), 1),
            })
    R = pd.DataFrame(rows)
    R.to_csv("results/transfer_cells.csv", index=False)

    for cond, g in R.groupby("condition"):
        print(f"\n=== {cond} : macro-F1, decontaminated (rows=train, cols=test) ===")
        M = g.pivot_table(index="train", columns="target", values="f1_decontam")
        M.columns = [c[:14] for c in M.columns]
        print(M.round(3).to_string())
        ind = g[g.in_domain].groupby("train").f1.mean()
        ood = g[~g.in_domain].groupby("train").f1_decontam.mean()
        gap = pd.DataFrame({"in_domain": ind, "ood_mean": ood, "delta_gen": ind - ood})
        print("\n" + gap.round(3).to_string())
        print(f"  MEAN in-domain={ind.mean():.3f}  ood={ood.mean():.3f}  Delta_gen={(ind - ood).mean():.3f}")

    # ---- the ACE ablation table: mean +/- std over seeds ------------------- #
    def agg(sub: pd.DataFrame, col: str) -> pd.DataFrame:
        per_seed = sub.groupby(["condition", "train", "seed"])[col].mean().reset_index()
        m = per_seed.pivot_table(index="train", columns="condition", values=col, aggfunc="mean")
        s = per_seed.pivot_table(index="train", columns="condition", values=col, aggfunc="std")
        # With a single seed the std is NaN and pivot_table drops the column
        # outright, which would blank the whole cell instead of just the +/-.
        # Reindex so a partially-completed array still reports its means.
        s = s.reindex(index=m.index, columns=m.columns).fillna(0.0)
        return m.round(3).astype(str) + s.round(3).map(lambda v: f" ±{v}")

    print("\n=== in-domain macro-F1 by condition (mean ± std over seeds) ===")
    print(agg(R[R.in_domain], "f1").to_string())
    print("\n=== decontaminated OOD mean macro-F1 by condition (mean ± std over seeds) ===")
    print(agg(R[~R.in_domain], "f1_decontam").to_string())

    # headline: does ACE trade in-domain for generalization?
    summ = []
    for cond, g in R.groupby("condition"):
        ps = g.groupby("seed").apply(
            lambda x: pd.Series({"ind": x[x.in_domain].f1.mean(),
                                 "ood": x[~x.in_domain].f1_decontam.mean()})
        )
        summ.append({"condition": cond, "n_seeds": len(ps),
                     "in_domain": ps.ind.mean(), "in_domain_std": ps.ind.std(),
                     "ood_decontam": ps.ood.mean(), "ood_std": ps.ood.std(),
                     "delta_gen": (ps.ind - ps.ood).mean()})
    S = pd.DataFrame(summ).set_index("condition")
    print("\n=== HEADLINE: aggregate by condition ===")
    print(S.round(3).to_string())
    S.to_csv("results/transfer_summary.csv")
    print("\nwrote results/transfer_cells.csv, results/transfer_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scientific_subsets")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only)")
    ap.add_argument("--task-id", type=int, default=None,
                    help="SLURM_ARRAY_TASK_ID: run only the i-th job of the "
                         "(condition x subset x seed) grid, so the array parallelises")
    ap.add_argument("--count-tasks", action="store_true",
                    help="print the grid size and exit (use for --array=0-N%K)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="validate the full train->eval->save path on N examples "
                         "(and N test docs/target) before submitting an array")
    ap.add_argument("--prefetch", action="store_true",
                    help="download the model on a login node before array launch")
    a = ap.parse_args()

    if a.prefetch:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        AutoTokenizer.from_pretrained(MODEL)
        AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        print(f"cached {MODEL}")
        return
    if a.prepare:
        prepare(Path(a.data))

    names = a.subsets or [f.stem for f in sorted(WORK.glob("*.parquet"))
                          if not f.stem.startswith("model_")]
    grid = [(c, n, s) for c in a.conditions for n in names for s in a.seeds]
    if a.count_tasks:
        print(len(grid))
        return

    if a.train:
        import torch
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        jobs = [grid[a.task_id]] if a.task_id is not None else grid
        for cond, n, s in jobs:
            try:
                train_one(n, cond, dev, seed=s, batch=a.batch, amp=a.amp, smoke=a.smoke)
            except Exception as e:                  # keep a long queue alive
                print(f"[FAIL] {n}/{cond}/s{s}: {type(e).__name__}: {e}", flush=True)
    if a.report:
        report()
    if not (a.prepare or a.train or a.report):
        ap.error("pass --prepare, --train and/or --report")


if __name__ == "__main__":
    main()
