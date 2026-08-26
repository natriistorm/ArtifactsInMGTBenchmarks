#!/usr/bin/env python3
"""
How much do the scientific-document ATD benchmarks overlap?
Usage
-----
    python contamination.py
    python contamination.py --cap 30000 --threshold 0.8 --out results/contamination

Outputs (under --out, default 'results/contamination'):
    _pairwise.csv    ordered pairs: % of row subset with a near-dup in col subset
    _matrix.csv      the same as a 10x10 matrix (for the heatmap)
    _intra.csv       per-subset internal exact- and near-duplicate rates
    _report.md       ready-to-paste summary with the worst pairs
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
N_PERM = 128
N_BANDS = 32
SHINGLE = 5
MERSENNE = (1 << 61) - 1
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def norm_text(t: str) -> str:
    """Aggressive normalization: overlap should be detected regardless of the
    whitespace/punctuation differences that Experiment B is about."""
    return _WS.sub(" ", _NONWORD.sub(" ", t.lower())).strip()


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf8"), digest_size=8).digest(), "big")


def shingle_hashes(t: str, k: int = SHINGLE) -> np.ndarray:
    w = norm_text(t).split()
    if len(w) < k:
        return np.array([_h64(" ".join(w))] if w else [0], dtype=np.uint64)
    return np.fromiter(
        (_h64(" ".join(w[i:i + k])) for i in range(len(w) - k + 1)),
        dtype=np.uint64,
        count=len(w) - k + 1,
    )


def signatures(texts: list[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n_docs, N_PERM) uint64 MinHash signatures."""
    sig = np.empty((len(texts), N_PERM), dtype=np.uint64)
    for i, t in enumerate(texts):
        h = shingle_hashes(t).astype(np.uint64)
        perm = (np.outer(a, h) + b[:, None]) % MERSENNE
        sig[i] = perm.min(axis=1)
    return sig


def band_keys(sig: np.ndarray) -> list[np.ndarray]:
    """One array of per-doc band hashes for each of N_BANDS bands."""
    rows = N_PERM // N_BANDS
    keys = []
    for bi in range(N_BANDS):
        block = sig[:, bi * rows:(bi + 1) * rows]
        keys.append(np.array([_h64(",".join(map(str, r))) for r in block], dtype=np.uint64))
    return keys


def build_index(keys: list[np.ndarray]) -> list[dict[int, list[int]]]:
    idx = []
    for k in keys:
        d: dict[int, list[int]] = defaultdict(list)
        for i, v in enumerate(k):
            d[int(v)].append(i)
        idx.append(d)
    return idx


def matched_mask(sig_a, keys_a, idx_b, sig_b, threshold: float, exclude_self: bool = False) -> np.ndarray:
    """Boolean mask over A: does each doc have >=1 doc in B with estimated
    Jaccard >= threshold? Computed once, then aggregated overall and per label."""
    n = sig_a.shape[0]
    out = np.zeros(n, dtype=bool)
    if n == 0 or sig_b.shape[0] == 0:
        return out
    for i in range(n):
        cand: set[int] = set()
        for bi in range(N_BANDS):
            cand.update(idx_b[bi].get(int(keys_a[bi][i]), ()))
        if exclude_self:
            cand.discard(i)
        if not cand:
            continue
        c = np.fromiter(cand, dtype=np.int64, count=len(cand))
        out[i] = bool((np.mean(sig_b[c] == sig_a[i], axis=1) >= threshold).any())
    return out


def pct(mask: np.ndarray) -> float:
    return 100.0 * float(mask.mean()) if mask.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scientific_subsets")
    ap.add_argument("--cap", type=int, default=30_000, help="max docs sampled per subset")
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard near-dup threshold")
    ap.add_argument("--out", default="results/contamination")
    args = ap.parse_args()

    d = Path(args.data)
    files = [
        f for f in sorted(d.glob("*.parquet"))
        if "sample3000" not in f.name and "scientific_all" not in f.name
    ]
    if not files:
        raise FileNotFoundError(f"no subset parquets under {d}")

    rng = np.random.default_rng(SEED)
    a = rng.integers(1, MERSENNE, size=N_PERM, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=N_PERM, dtype=np.uint64)

    data, sigs, keys, idxs, exact = {}, {}, {}, {}, {}
    intra_rows = []

    for fp in files:
        name = fp.stem
        df = pd.read_parquet(fp, columns=["text", "label"])
        if len(df) > args.cap:
            df = df.sample(args.cap, random_state=SEED).reset_index(drop=True)
        print(f"[minhash] {name:24s} n={len(df)}", flush=True)
        data[name] = df
        sigs[name] = signatures(df.text.tolist(), a, b)
        keys[name] = band_keys(sigs[name])
        idxs[name] = build_index(keys[name])
        exact[name] = np.array(
            [_h64(norm_text(t)) for t in df.text], dtype=np.uint64
        )

        _, counts = np.unique(exact[name], return_counts=True)
        exact_dup = 100.0 * (len(exact[name]) - len(counts)) / len(exact[name])
        near = pct(matched_mask(
            sigs[name], keys[name], idxs[name], sigs[name], args.threshold, exclude_self=True
        ))
        intra_rows.append({
            "subset": name, "n": len(df),
            "exact_dup_pct": round(exact_dup, 2),
            "near_dup_pct": round(near, 2),
        })
        print(f"           internal: exact={exact_dup:.2f}%  near={near:.2f}%", flush=True)

    names = list(data)
    pair_rows = []
    for A in names:
        setB_exact = {n: set(exact[n].tolist()) for n in names}
        for B in names:
            if A == B:
                pair_rows.append({
                    "row_subset": A, "col_subset": B,
                    "near_dup_pct": 100.0, "exact_dup_pct": 100.0,
                    "near_dup_human_pct": 100.0, "near_dup_machine_pct": 100.0,
                })
                continue
            mask = matched_mask(sigs[A], keys[A], idxs[B], sigs[B], args.threshold)
            ex = 100.0 * np.mean([h in setB_exact[B] for h in exact[A].tolist()])
            lab_a = data[A].label.to_numpy()
            near = pct(mask)
            per_label = {
                tag: pct(mask[lab_a == lab]) if (lab_a == lab).any() else float("nan")
                for lab, tag in [(0, "human"), (1, "machine")]
            }
            pair_rows.append({
                "row_subset": A, "col_subset": B,
                "near_dup_pct": round(near, 2), "exact_dup_pct": round(ex, 2),
                "near_dup_human_pct": round(per_label["human"], 2),
                "near_dup_machine_pct": round(per_label["machine"], 2),
            })
            if near > 0.5:
                print(f"[overlap] {A} -> {B}: near={near:.1f}% exact={ex:.1f}% "
                      f"(human={per_label['human']:.1f}% machine={per_label['machine']:.1f}%)",
                      flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    P = pd.DataFrame(pair_rows)
    P.to_csv(f"{out}_pairwise.csv", index=False)
    M = P.pivot(index="row_subset", columns="col_subset", values="near_dup_pct")
    M.to_csv(f"{out}_matrix.csv")
    I = pd.DataFrame(intra_rows)
    I.to_csv(f"{out}_intra.csv", index=False)

    worst = (
        P[P.row_subset != P.col_subset]
        .sort_values("near_dup_pct", ascending=False)
        .head(10)
    )
    md = [
        f"# Contamination (Jaccard >= {args.threshold}, {SHINGLE}-word shingles, "
        f"{N_PERM}-perm MinHash, cap {args.cap}/subset)\n",
        "## Worst overlapping pairs (% of row subset found in col subset)\n",
        "| A | B | near-dup | exact | human | machine |", "|---|---|---|---|---|---|",
    ]
    for r in worst.itertuples():
        md.append(f"| {r.row_subset} | {r.col_subset} | {r.near_dup_pct:.1f}% | "
                  f"{r.exact_dup_pct:.1f}% | {r.near_dup_human_pct:.1f}% | {r.near_dup_machine_pct:.1f}% |")
    md += ["\n## Internal duplication\n", "| Subset | n | exact | near |", "|---|---|---|---|"]
    for r in I.itertuples():
        md.append(f"| {r.subset} | {r.n} | {r.exact_dup_pct:.2f}% | {r.near_dup_pct:.2f}% |")
    md += ["\n## Full matrix (near-dup %)\n", M.round(1).to_markdown()]
    Path(f"{out}_report.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out}_pairwise.csv, {out}_matrix.csv, {out}_intra.csv, {out}_report.md")


if __name__ == "__main__":
    main()
