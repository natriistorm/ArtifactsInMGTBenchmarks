#!/usr/bin/env python3
"""
isolate_scientific_subsets.py

Isolate the *scientific* subsets from the ATD datasets already used in the paper
and normalize them to one schema for downstream analysis (PHD / JSD_TTS /
perturbation / detector baselines).

What it produces
----------------
For every configured dataset it keeps only scientific-domain rows and writes a
unified table with columns:

    text, label, label_str, domain, dataset, vintage, generator, n_tokens, n_chars

    label     : 0 = human, 1 = machine
    domain    : 'arxiv' (scientific abstracts) or 'peerread' (peer reviews),
                or 'sci_paper' for IDMGSP introductions
    vintage   : free tag you assign per source (e.g. m4 / semeval24 / coling25)
                so the generator-vintage comparison is available for free

Two dataset "kinds" are handled:
  * m4_family : M4 / SemEval-2024 Task 8 / COLING-2025 Task 1 (and anything with
                a text/label/source schema). Rows are filtered by a normalized
                domain column to {arxiv, peerread}.
  * idmgsp    : IDMGSP. The introduction field is used as the text (as in the
                paper); abstracts are too short.

The script is deliberately defensive: it *auto-detects* the text/label/domain/
generator columns from a list of candidates and prints what it matched, so you
can confirm the isolation is right before trusting the output. Edit the DATASETS
list below to point at your local files.

Nothing here forces class balance — you sample 1000 balanced per class later.
It only isolates, normalizes, and reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIG — edit paths to your local files, then run.                          #
# --------------------------------------------------------------------------- #

@dataclass
class DatasetSpec:
    name: str                       # short name, used in the `dataset` column
    kind: str                       # 'm4_family' or 'idmgsp'
    path: str                       # file OR directory (dir => read all files in it)
    vintage: str                    # free tag: m4 / semeval24 / coling25 / idmgsp
    # optional explicit column overrides (leave None to auto-detect)
    text_col: str | None = None
    label_col: str | None = None
    domain_col: str | None = None
    generator_col: str | None = None
    # idmgsp only: which field to use as text
    idmgsp_text_field: str = "introduction"


DATASETS: list[DatasetSpec] = [
    # --- M4 GitHub layout: per-file human_text/machine_text pairs, domain from filename.
    DatasetSpec(name="M4",         kind="m4_paired", path="M4/data",                          vintage="m4"),
    # --- SemEval-2024 Task 8: text/label/source schema, domain in `source`.
    DatasetSpec(name="SemEval24",  kind="m4_family", path="subtaskA_train_monolingual.jsonl",  vintage="semeval24"),
    # --- COLING-2025 Task 1: domain lives in `sub_source`, NOT `source`
    #     (`source` here is the upstream corpus tag, e.g. "m4gt"/"mage" — not a domain).
    DatasetSpec(name="COLING25",   kind="m4_family", path="coling.jsonl",                      vintage="coling25",
                domain_col="sub_source"),
    # --- IDMGSP: scientific paper introductions ---------------------------- #
    DatasetSpec(name="IDMGSP",     kind="idmgsp",    path="idmgsp_full.csv",                   vintage="idmgsp"),
    # --- CheckGPT: CS / Physics abstracts, ground.json (human) vs GPT-WRI generations
    #     (gpt_task1_prompt{1..4}.json — write-from-scratch given only the title; task2/3
    #     "complete"/"polish" are excluded). Whole corpus is scientific abstracts already,
    #     so there is no domain filter step here.
    DatasetSpec(name="CheckGPT",   kind="checkgpt",  path="checkgpt_v2",                       vintage="checkgpt_v2"),
    # --- CHEAT: IEEE abstracts, ieee-init.jsonl (human) vs ieee-chatgpt-generation.jsonl
    #     (write-from-scratch, same GPT-WRI convention as CheckGPT). ieee-chatgpt-polish
    #     (rewrite of the human abstract) and ieee-chatgpt-fusion (human/AI hybrid) are
    #     intentionally excluded for the same reason CheckGPT's task2/3 were.
    DatasetSpec(name="CHEAT",      kind="cheat",     path="CHEAT/data",                        vintage="cheat"),
]

# CheckGPT sub-folder -> canonical domain name for the `domain` column.
CHECKGPT_DOMAINS: dict[str, str] = {"CS": "cs_paper", "PHX": "physics_paper"}

# Target scientific domains for the M4 family. Raw values are lowercased and
# matched against these tokens by substring. NOTE: bare "review" is intentionally
# NOT here — in M4 the peer-review domain is tagged "peerread"; "review" alone
# would wrongly pull in product/Amazon reviews.
DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "arxiv":    ("arxiv", "arxiv_abstract", "arxiv-abstract"),
    "peerread": ("peerread", "peer_read", "peer-read", "peerread_review"),
}

# Column auto-detection candidates (checked in order, case-insensitive).
TEXT_CANDIDATES     = ("text", "content", "document", "review", "body", "generation")
LABEL_CANDIDATES    = ("label", "labels", "class", "target", "is_generated", "generated", "machine")
DOMAIN_CANDIDATES   = ("source", "domain", "domain_name", "subsource", "category", "genre")
GENERATOR_CANDIDATES = ("model", "generator", "source_model", "llm", "model_name")

# Label token normalization -> 0 (human) / 1 (machine).
HUMAN_TOKENS   = {"0", "human", "h", "real", "original", "gold", "authentic"}
MACHINE_TOKENS = {"1", "machine", "ai", "generated", "fake", "synthetic", "gpt", "llm", "bot"}

# PHD / JSD_TTS reliability: these topological metrics need long texts. The paper
# excluded RuATD (median ~99 tokens) and AuTex (~386) as too short. Flag a subset
# as PHD-unreliable if its median whitespace-token count is below this.
MIN_TOKENS_PHD = 250

# Drop obviously degenerate rows below this many tokens (empty / truncated).
MIN_TOKENS_KEEP = 5

# Balanced sample written per (dataset, domain) subset, e.g. *_sample3000.parquet.
SAMPLE_PER_CLASS = 1500
SAMPLE_SEED = 42

READABLE_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".parquet", ".pq"}


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #

def _read_one(fp: Path) -> pd.DataFrame:
    """Read a single file into a DataFrame, dispatching on suffix."""
    suf = fp.suffix.lower()
    if suf == ".jsonl":
        return pd.read_json(fp, lines=True)
    if suf == ".json":
        # could be a records list or a single object
        try:
            return pd.read_json(fp)
        except ValueError:
            return pd.read_json(fp, lines=True)
    if suf == ".csv":
        return pd.read_csv(fp)
    if suf == ".tsv":
        return pd.read_csv(fp, sep="\t")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(fp)
    raise ValueError(f"Unsupported file type: {fp}")


def load_frame(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load a file or every readable file in a directory. Returns (df, filenames).

    The originating filename is kept in a `_src_file` column — useful for the
    per-file M4 GitHub layout (e.g. arxiv_chatgpt.jsonl) where the domain and
    generator live in the filename rather than a column.
    """
    p = Path(path)
    files: list[Path]
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in READABLE_SUFFIXES)
    elif p.exists():
        files = [p]
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    if not files:
        raise FileNotFoundError(f"No readable files ({READABLE_SUFFIXES}) under: {path}")

    frames = []
    for f in files:
        df = _read_one(f)
        df["_src_file"] = f.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out, [f.name for f in files]


def _find_name(names, candidates, override: str | None) -> str | None:
    """Match a candidate column/key name (case-insensitive) against `names`."""
    names = list(names)
    if override:
        if override in names:
            return override
        print(f"    ! override column '{override}' not found; falling back to auto-detect")
    lower = {str(n).lower(): n for n in names}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _find_col(df: pd.DataFrame, candidates, override: str | None) -> str | None:
    return _find_name(df.columns, candidates, override)


# --------------------------------------------------------------------------- #
# Normalization                                                               #
# --------------------------------------------------------------------------- #

def normalize_domain(raw) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()
    for canon, aliases in DOMAIN_ALIASES.items():
        if any(a in s for a in aliases):
            return canon
    return None


def normalize_label(raw) -> int | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()
    if s in HUMAN_TOKENS:
        return 0
    if s in MACHINE_TOKENS:
        return 1
    # numeric fallback: anything != 0 is treated as machine
    try:
        return 0 if float(s) == 0.0 else 1
    except ValueError:
        # unknown string token — likely a generator name in a label col => machine
        return 1


def add_length_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_chars"] = df["text"].str.len()
    df["n_tokens"] = df["text"].str.split().map(len)
    return df


# --------------------------------------------------------------------------- #
# Per-kind loaders                                                            #
# --------------------------------------------------------------------------- #

def _stream_jsonl_scientific(fp: Path, spec: DatasetSpec) -> pd.DataFrame:
    """Stream a single large .jsonl file line-by-line, keeping only rows whose
    domain matches DOMAIN_ALIASES. Avoids ever materializing the full file
    (hundreds of MB / 600k+ rows) as one DataFrame just to throw most of it away.
    """
    with fp.open() as fh:
        first = json.loads(fh.readline())
    keys = list(first.keys())

    text_col = _find_name(keys, TEXT_CANDIDATES, spec.text_col)
    label_col = _find_name(keys, LABEL_CANDIDATES, spec.label_col)
    domain_col = _find_name(keys, DOMAIN_CANDIDATES, spec.domain_col)
    gen_col = _find_name(keys, GENERATOR_CANDIDATES, spec.generator_col)
    print(f"  detected keys -> text={text_col!r} label={label_col!r} "
          f"domain={domain_col!r} generator={gen_col!r}")
    if text_col is None or label_col is None or domain_col is None:
        raise ValueError(
            f"[{spec.name}] could not find text/label/domain keys. "
            f"Keys present: {keys}. Set text_col/label_col/domain_col in the spec."
        )

    kept = []
    n_seen = 0
    with fp.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            d = json.loads(line)
            domain = normalize_domain(d.get(domain_col))
            if domain is None:
                continue
            kept.append({
                "text": str(d.get(text_col)),
                "label": normalize_label(d.get(label_col)),
                "domain": domain,
                "generator": str(d.get(gen_col)) if gen_col else pd.NA,
            })
    print(f"  streamed {n_seen:,} rows from {fp.name} -> {len(kept):,} scientific-domain rows")

    df = pd.DataFrame(kept)
    df["dataset"] = spec.name
    df["vintage"] = spec.vintage
    return df


def process_m4_family(spec: DatasetSpec) -> pd.DataFrame:
    p = Path(spec.path)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        # Single large jsonl (SemEval/COLING releases): filter while streaming
        # instead of concatenating the whole file into memory first.
        df = _stream_jsonl_scientific(p, spec)
        if df["label"].isna().any():
            n_bad = int(df["label"].isna().sum())
            print(f"  ! dropping {n_bad} rows with unmappable label")
            df = df[df["label"].notna()]
        df["label"] = df["label"].astype(int)
        return df

    raw, files = load_frame(spec.path)
    print(f"  loaded {len(raw):,} rows from {len(files)} file(s): {files[:4]}{'...' if len(files) > 4 else ''}")

    text_col = _find_col(raw, TEXT_CANDIDATES, spec.text_col)
    label_col = _find_col(raw, LABEL_CANDIDATES, spec.label_col)
    domain_col = _find_col(raw, DOMAIN_CANDIDATES, spec.domain_col)
    gen_col = _find_col(raw, GENERATOR_CANDIDATES, spec.generator_col)
    print(f"  detected columns -> text={text_col!r} label={label_col!r} "
          f"domain={domain_col!r} generator={gen_col!r}")

    if text_col is None or label_col is None:
        raise ValueError(
            f"[{spec.name}] could not find text/label columns. "
            f"Columns present: {list(raw.columns)}. Set text_col/label_col in the spec."
        )

    # Domain source: a column if present, else parse from the filename (per-file layout).
    if domain_col is not None:
        domain_raw = raw[domain_col]
    else:
        print("  ! no domain column found — deriving domain from filename (_src_file)")
        domain_raw = raw["_src_file"]

    df = pd.DataFrame({
        "text": raw[text_col].astype(str),
        "label": raw[label_col].map(normalize_label),
        "domain": domain_raw.map(normalize_domain),
        "generator": raw[gen_col].astype(str) if gen_col else pd.NA,
        "dataset": spec.name,
        "vintage": spec.vintage,
    })

    before = len(df)
    df = df[df["domain"].notna()].copy()          # keep only arxiv / peerread
    print(f"  scientific-domain rows: {len(df):,} / {before:,} "
          f"({df['domain'].value_counts().to_dict()})")

    if df["label"].isna().any():
        n_bad = int(df["label"].isna().sum())
        print(f"  ! dropping {n_bad} rows with unmappable label")
        df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)
    return df


def process_m4_paired(spec: DatasetSpec) -> pd.DataFrame:
    """M4 GitHub layout: one file per (domain, generator), each row holding a
    paired human_text/machine_text (not a single text/label schema). Only
    arxiv_*/peerread_* files are read — the other ~25 domain files (reddit,
    wikipedia, wikihow, ...) are skipped before ever being opened.
    """
    p = Path(spec.path)
    if not p.is_dir():
        raise FileNotFoundError(f"[{spec.name}] expected a directory, got: {p}")

    files = sorted(
        f for f in p.glob("*.jsonl")
        if normalize_domain(f.name) is not None
    )
    if not files:
        raise FileNotFoundError(f"[{spec.name}] no arxiv/peerread .jsonl files under: {p}")

    parts = []
    for f in files:
        domain = normalize_domain(f.name)
        rows = [json.loads(line) for line in f.open() if line.strip()]
        human_col = _find_name(rows[0].keys(), ("human_text", "text"), spec.text_col)
        machine_col = _find_name(rows[0].keys(), ("machine_text",), None)
        model_col = _find_name(rows[0].keys(), GENERATOR_CANDIDATES, spec.generator_col)
        if human_col is None or machine_col is None:
            print(f"  ! skipping {f.name}: could not find human/machine text keys "
                  f"({list(rows[0].keys())})")
            continue

        human = pd.DataFrame({
            "text": [str(r.get(human_col)) for r in rows],
            "label": 0,
            "domain": domain,
            "generator": "human",
        })
        machine = pd.DataFrame({
            "text": [str(r.get(machine_col)) for r in rows],
            "label": 1,
            "domain": domain,
            "generator": [str(r.get(model_col)) for r in rows] if model_col else "unknown",
        })
        part = pd.concat([human, machine], ignore_index=True)
        print(f"  {f.name}: {len(rows):,} pairs -> {len(part):,} rows (domain={domain})")
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df["dataset"] = spec.name
    df["vintage"] = spec.vintage
    return df


def process_idmgsp(spec: DatasetSpec) -> pd.DataFrame:
    raw, files = load_frame(spec.path)
    print(f"  loaded {len(raw):,} rows from {len(files)} file(s): {files[:4]}{'...' if len(files) > 4 else ''}")

    field = spec.idmgsp_text_field
    if field not in raw.columns:
        # be forgiving about capitalization / spacing
        lower = {c.lower(): c for c in raw.columns}
        if field.lower() in lower:
            field = lower[field.lower()]
        else:
            raise ValueError(
                f"[{spec.name}] introduction field '{spec.idmgsp_text_field}' not found. "
                f"Columns: {list(raw.columns)}. Set idmgsp_text_field in the spec."
            )

    label_col = _find_col(raw, LABEL_CANDIDATES, spec.label_col)
    if label_col is None:
        raise ValueError(
            f"[{spec.name}] could not find a label column. Columns: {list(raw.columns)}."
        )
    print(f"  detected columns -> text={field!r} label={label_col!r}")

    df = pd.DataFrame({
        "text": raw[field].astype(str),
        "label": raw[label_col].map(normalize_label).astype("Int64"),
        "domain": "sci_paper",
        "generator": pd.NA,
        "dataset": spec.name,
        "vintage": spec.vintage,
    })
    # drop rows whose introduction is empty / missing
    df = df[df["text"].str.strip().ne("") & df["text"].str.lower().ne("nan")]
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    print(f"  usable introduction rows: {len(df):,} "
          f"(labels: {df['label'].value_counts().to_dict()})")
    return df


def _load_checkgpt_json(fp: Path) -> dict:
    """gpt_task*/ground files are {"0": {...}, "1": {...}, ...} keyed by shared
    numeric-string id, one dict per abstract. json.load (not pd.read_json) so
    the numeric keys never get reinterpreted as columns.
    """
    with fp.open() as fh:
        return json.load(fh)


def process_checkgpt(spec: DatasetSpec) -> pd.DataFrame:
    root = Path(spec.path)
    if not root.is_dir():
        raise FileNotFoundError(f"[{spec.name}] expected a directory, got: {root}")

    domain_dirs = [root / d for d in CHECKGPT_DOMAINS if (root / d).is_dir()]
    if not domain_dirs:
        raise FileNotFoundError(
            f"[{spec.name}] none of {list(CHECKGPT_DOMAINS)} found under: {root}"
        )

    parts = []
    for ddir in domain_dirs:
        domain = CHECKGPT_DOMAINS[ddir.name]

        ground_fp = ddir / "ground.json"
        if not ground_fp.exists():
            print(f"  ! skipping {ddir.name}: no ground.json")
            continue
        ground = _load_checkgpt_json(ground_fp)
        human = pd.DataFrame({
            "text": [str(v.get("abstract")) for v in ground.values()],
            "label": 0,
            "domain": domain,
            "generator": "human",
        })
        print(f"  {ddir.name}/ground.json: {len(human):,} human rows")
        parts.append(human)

        # GPT-WRI only: task1 = write the abstract from scratch given just the
        # title. task2 ("complete", continues a truncated human abstract) and
        # task3 ("polish", rewrites a full human abstract) are intentionally
        # excluded — they aren't from-scratch generation.
        for f in sorted(ddir.glob("gpt_task1_prompt*.json")):
            data = _load_checkgpt_json(f)
            machine = pd.DataFrame({
                "text": [str(v.get("abstract")) for v in data.values()],
                "label": 1,
                "domain": domain,
                "generator": f.stem.removeprefix("gpt_"),   # e.g. "task1_prompt1"
            })
            print(f"  {ddir.name}/{f.name}: {len(machine):,} machine rows")
            parts.append(machine)

    df = pd.concat(parts, ignore_index=True)
    df["dataset"] = spec.name
    df["vintage"] = spec.vintage
    return df


def process_cheat(spec: DatasetSpec) -> pd.DataFrame:
    """CHEAT: ieee-init.jsonl (human) vs ieee-chatgpt-generation.jsonl (GPT-WRI,
    write-from-scratch), row-aligned by shared `id`. polish/fusion excluded.
    """
    root = Path(spec.path)
    init_fp = root / "ieee-init.jsonl"
    gen_fp = root / "ieee-chatgpt-generation.jsonl"
    if not init_fp.exists() or not gen_fp.exists():
        raise FileNotFoundError(f"[{spec.name}] expected {init_fp.name} and {gen_fp.name} under: {root}")

    init_rows = [json.loads(line) for line in init_fp.open() if line.strip()]
    gen_rows = [json.loads(line) for line in gen_fp.open() if line.strip()]

    human = pd.DataFrame({
        "text": [str(r.get("abstract")) for r in init_rows],
        "label": 0,
        "domain": "ieee_paper",
        "generator": "human",
    })
    print(f"  {init_fp.name}: {len(human):,} human rows")

    machine = pd.DataFrame({
        "text": [str(r.get("abstract")) for r in gen_rows],
        "label": 1,
        "domain": "ieee_paper",
        "generator": "chatgpt_generation",
    })
    print(f"  {gen_fp.name}: {len(machine):,} machine rows")

    df = pd.concat([human, machine], ignore_index=True)
    df["dataset"] = spec.name
    df["vintage"] = spec.vintage
    return df


KIND_DISPATCH = {
    "m4_family": process_m4_family,
    "m4_paired": process_m4_paired,
    "idmgsp": process_idmgsp,
    "checkgpt": process_checkgpt,
    "cheat": process_cheat,
}


def sample_balanced(g: pd.DataFrame, n_per_class: int, seed: int = SAMPLE_SEED) -> pd.DataFrame | None:
    """n_per_class human + n_per_class machine rows, drawn only from rows with
    >= MIN_TOKENS_KEEP tokens. Returns None if either class doesn't have enough
    usable rows.
    """
    pool = g[g["n_tokens"] >= MIN_TOKENS_KEEP]
    picks = []
    for lbl in (0, 1):
        cand = pool[pool["label"] == lbl]
        if len(cand) < n_per_class:
            return None
        picks.append(cand.sample(n=n_per_class, random_state=seed))
    return pd.concat(picks, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def build_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, vintage, domain), g in df.groupby(["dataset", "vintage", "domain"], dropna=False):
        n_h = int((g["label"] == 0).sum())
        n_m = int((g["label"] == 1).sum())
        med = float(g["n_tokens"].median())
        rows.append({
            "dataset": dataset,
            "vintage": vintage,
            "domain": domain,
            "n_total": len(g),
            "n_human": n_h,
            "n_machine": n_m,
            "median_tokens": round(med, 1),
            "mean_tokens": round(float(g["n_tokens"].mean()), 1),
            "median_tokens_human": round(float(g.loc[g.label == 0, "n_tokens"].median()), 1) if n_h else None,
            "median_tokens_machine": round(float(g.loc[g.label == 1, "n_tokens"].median()), 1) if n_m else None,
            "PHD_reliable": "yes" if med >= MIN_TOKENS_PHD else "NO (too short)",
        })
    return pd.DataFrame(rows).sort_values(["domain", "vintage", "dataset"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Isolate scientific subsets for ATD dataset-quality analysis.")
    ap.add_argument("--out", default="scientific_subsets", help="output directory")
    ap.add_argument("--drop-short", action="store_true",
                    help=f"drop rows with < {MIN_TOKENS_KEEP} tokens (default: keep, just report)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_parts = []
    all_samples = []
    for spec in DATASETS:
        print(f"\n=== {spec.name} ({spec.kind}, vintage={spec.vintage}) ===")
        try:
            part = KIND_DISPATCH[spec.kind](spec)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIPPED: {e}")
            continue
        if part.empty:
            print("  SKIPPED: no rows survived isolation.")
            continue

        part = add_length_cols(part)

        n_deg = int((part["n_tokens"] < MIN_TOKENS_KEEP).sum())
        if n_deg:
            msg = f"  {n_deg} rows below {MIN_TOKENS_KEEP} tokens"
            if args.drop_short:
                part = part[part["n_tokens"] >= MIN_TOKENS_KEEP]
                print(msg + " -> dropped")
            else:
                print(msg + " -> kept (use --drop-short to remove)")

        # write one parquet per (dataset, domain), plus a balanced sample
        for domain, g in part.groupby("domain"):
            fp = out_dir / f"{spec.name}_{domain}.parquet"
            g.reset_index(drop=True).to_parquet(fp, index=False)
            print(f"  wrote {fp}  ({len(g):,} rows)")

            sampled = sample_balanced(g, SAMPLE_PER_CLASS)
            if sampled is None:
                n_h = int((g["label"] == 0).sum())
                n_m = int((g["label"] == 1).sum())
                print(f"  ! {spec.name}_{domain}: not enough usable rows for a "
                      f"{SAMPLE_PER_CLASS}/{SAMPLE_PER_CLASS} balanced sample "
                      f"(have {n_h} human / {n_m} machine) -> skipped")
            else:
                sample_fp = out_dir / f"{spec.name}_{domain}_sample{2 * SAMPLE_PER_CLASS}.parquet"
                sampled.to_parquet(sample_fp, index=False)
                print(f"  wrote {sample_fp}  ({len(sampled):,} rows: "
                      f"{SAMPLE_PER_CLASS} human / {SAMPLE_PER_CLASS} AI)")
                all_samples.append(sampled)
        all_parts.append(part)

    if not all_parts:
        print("\nNo data produced. Check the paths in the DATASETS config.", file=sys.stderr)
        return 1

    combined = pd.concat(all_parts, ignore_index=True)
    combined_fp = out_dir / "scientific_all.parquet"
    combined.to_parquet(combined_fp, index=False)

    if all_samples:
        combined_sample = pd.concat(all_samples, ignore_index=True)
        combined_sample_fp = out_dir / "scientific_all_sampled.parquet"
        combined_sample.to_parquet(combined_sample_fp, index=False)
        print(f"\ncombined balanced sample -> {combined_sample_fp}  ({len(combined_sample):,} rows)")

    report = build_report(combined)
    report_fp = out_dir / "subset_report.csv"
    report.to_csv(report_fp, index=False)

    print("\n" + "=" * 78)
    print("SUMMARY  (PHD/JSD_TTS need long texts; short subsets are perturbation+detector only)")
    print("=" * 78)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(report.to_string(index=False))
    print(f"\ncombined -> {combined_fp}")
    print(f"report   -> {report_fp}")
    print("\nSanity check: eyeball a couple of rows per label to confirm the "
          "human/machine mapping is correct for each source:")
    for lbl in (0, 1):
        ex = combined[combined.label == lbl]
        if not ex.empty:
            s = ex.iloc[0]
            preview = s["text"][:160].replace("\n", " ")
            print(f"  label={lbl} [{s['dataset']}/{s['domain']}]: {preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())