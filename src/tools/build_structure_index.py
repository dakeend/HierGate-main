#!/usr/bin/env python3
"""Build a mutation-level index of Rosetta FastRelax WT/Mut PDB files.

The index is the download manifest requested by Reviewer 1, comment 6.
Each row is keyed by dataset / protein / chain / mutation and points to the
relaxed wild-type and mutant PDB files used for graph construction.

Example:
    python src/tools/build_structure_index.py \\
        --pdb-root /media/ST-18T/nianwen/origin-ucl/data/pdbs \\
        --name-root data/R3_names \\
        --out data/revision/structure_release/structure_index.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z]+)$", re.I)

OFFICIAL_SPLITS = [
    "s2648",
    "test",
    "myoglobin",
    "p53",
    "S250",
    "S350",
    "S605",
    "S879",
    "S1925",
    "S669",
]

SPLIT_ALIASES = {
    "s2648": ["s2648", "S2648"],
    "test": ["test", "Ssym", "ssym"],
    "myoglobin": ["myoglobin", "Myoglobin"],
    "p53": ["p53"],
    "S250": ["S250"],
    "S350": ["S350"],
    "S605": ["S605"],
    "S879": ["S879"],
    "S1925": ["S1925"],
    "S669": ["S669"],
}

NAME_FILE = {
    "s2648": "s2648_names.txt",
    "test": "test_names.txt",
    "myoglobin": "myoglobin_names.txt",
    "p53": "p53_names.txt",
    "S250": "S250_names.txt",
    "S350": "S350_names.txt",
    "S605": "S605_names.txt",
    "S879": "S879_names.txt",
    "S1925": "S1925_names.txt",
    "S669": "S669_names.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--name-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=OFFICIAL_SPLITS)
    parser.add_argument("--file-manifest", type=Path, default=None, help="Optional per-file listing.")
    parser.add_argument(
        "--parse-pdb",
        action="store_true",
        help="Read CA residue ids from each PDB (slower; used for site-mapping checks).",
    )
    parser.add_argument(
        "--include-residue-maps",
        action="store_true",
        help="Write full CA residue-id lists. Requires --parse-pdb.",
    )
    return parser.parse_args()


def parse_sample_id(sample_id: str) -> dict:
    protein, mut = sample_id.split("_", 1) if "_" in sample_id else (sample_id, "")
    match = MUT_RE.match(mut)
    wt = match.group(1).upper() if match else ""
    pos = match.group(2) if match else ""
    mut_aa = match.group(3).upper() if match else ""
    return {
        "sample_id": sample_id,
        "protein": protein,
        "pdb_id": protein[:4],
        "chain": protein[4:],
        "mutation": f"{wt}{pos}{mut_aa}" if wt else mut,
        "wt_aa": wt,
        "pdb_seqpos": pos,
        "mut_aa": mut_aa,
    }


def candidate_dirs(pdb_root: Path, split: str) -> list[Path]:
    dirs = []
    for name in SPLIT_ALIASES.get(split, [split]):
        path = pdb_root / name
        if path.is_dir():
            dirs.append(path)
    return dirs


def index_relaxed_files(split_dir: Path) -> dict[str, Path]:
    """One directory walk; keys are lowercase filenames."""
    found: dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(split_dir):
        for name in filenames:
            if name.endswith("_relaxed.pdb"):
                found[name.lower()] = Path(dirpath) / name
    return found


def relpath(pdb_root: Path, path: Path) -> str:
    return str(path.relative_to(pdb_root)).replace("\\", "/")


def resolve_pair(pdb_root: Path, file_index: dict[str, Path], protein: str, mutation: str) -> tuple[str, str, str]:
    wt_names = [
        f"{protein}_relaxed.pdb",
        f"{protein.lower()}_relaxed.pdb",
        f"{protein.upper()}_relaxed.pdb",
    ]
    mut_names = [
        f"{protein}_{mutation}_relaxed.pdb",
        f"{protein.lower()}_{mutation}_relaxed.pdb",
        f"{protein.upper()}_{mutation}_relaxed.pdb",
        f"{protein}_{mutation.upper()}_relaxed.pdb",
        f"{protein.lower()}_{mutation.upper()}_relaxed.pdb",
    ]
    wt = next((file_index[n.lower()] for n in wt_names if n.lower() in file_index), None)
    mut = next((file_index[n.lower()] for n in mut_names if n.lower() in file_index), None)
    wt_rel = relpath(pdb_root, wt) if wt else ""
    mut_rel = relpath(pdb_root, mut) if mut else ""
    if wt and mut:
        status = "wt_mut"
    elif wt:
        status = "wt_only"
    elif mut:
        status = "mut_only"
    else:
        status = "missing"
    return wt_rel, mut_rel, status


def load_sample_ids(name_root: Path | None, split: str, file_index: dict[str, Path]) -> list[str]:
    if name_root is not None:
        fname = NAME_FILE.get(split, f"{split}_names.txt")
        path = name_root / fname
        if path.exists():
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    ids = []
    mut_file_re = re.compile(r"^(.+)_([A-Za-z]\d+[A-Za-z]+)_relaxed\.pdb$")
    for name in file_index:
        match = mut_file_re.match(Path(name).name)
        if not match:
            continue
        sample_id = f"{match.group(1)}_{match.group(2)}"
        # recover original casing from the path name if possible
        actual = file_index[name].name
        match = mut_file_re.match(actual)
        if match:
            sample_id = f"{match.group(1)}_{match.group(2)}"
        if sample_id not in seen:
            seen.add(sample_id)
            ids.append(sample_id)
    return ids


def ca_residue_ids(pdb_path: Path) -> list[str]:
    resids = []
    try:
        with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21].strip()
                    resid = line[22:26].strip()
                    icode = line[26].strip() if len(line) > 26 else ""
                    resids.append(f"{chain}:{resid}{icode}")
    except OSError:
        return []
    return resids


def main() -> None:
    args = parse_args()
    rows = []
    file_rows = []
    for split in args.splits:
        split_dirs = candidate_dirs(args.pdb_root, split)
        if not split_dirs:
            print(f"[skip] no PDB directory for {split}", flush=True)
            continue
        file_index: dict[str, Path] = {}
        for split_dir in split_dirs:
            print(f"Indexing files under {split_dir} ...", flush=True)
            file_index.update(index_relaxed_files(split_dir))
        print(f"{split}: {len(file_index)} relaxed PDB files", flush=True)
        sample_ids = load_sample_ids(args.name_root, split, file_index)
        n_ok = 0
        for sample_id in sample_ids:
            meta = parse_sample_id(sample_id)
            wt_rel, mut_rel, status = resolve_pair(args.pdb_root, file_index, meta["protein"], meta["mutation"])
            n_wt = 0
            n_mut = 0
            wt_has_site = ""
            mut_has_site = ""
            wt_map = ""
            mut_map = ""
            site_token = f"{meta['chain']}:{meta['pdb_seqpos']}" if meta["chain"] and meta["pdb_seqpos"] else ""
            parse = args.parse_pdb or args.include_residue_maps
            if wt_rel:
                if parse:
                    wt_ids = ca_residue_ids(args.pdb_root / wt_rel)
                    n_wt = len(wt_ids)
                    wt_has_site = "yes" if site_token and any(tok.startswith(site_token) for tok in wt_ids) else "no"
                    if args.include_residue_maps:
                        wt_map = ";".join(wt_ids)
                file_rows.append({"dataset": split, "sample_id": sample_id, "role": "WT", "relpath": wt_rel, "n_ca": n_wt})
            if mut_rel:
                if parse:
                    mut_ids = ca_residue_ids(args.pdb_root / mut_rel)
                    n_mut = len(mut_ids)
                    mut_has_site = "yes" if site_token and any(tok.startswith(site_token) for tok in mut_ids) else "no"
                    if args.include_residue_maps:
                        mut_map = ";".join(mut_ids)
                file_rows.append({"dataset": split, "sample_id": sample_id, "role": "Mut", "relpath": mut_rel, "n_ca": n_mut})
            if status == "wt_mut":
                n_ok += 1
            row = {
                "dataset": split,
                "sample_id": sample_id,
                "protein": meta["protein"],
                "pdb_id": meta["pdb_id"],
                "chain": meta["chain"],
                "mutation": meta["mutation"],
                "wt_aa": meta["wt_aa"],
                "pdb_seqpos": meta["pdb_seqpos"],
                "mut_aa": meta["mut_aa"],
                "wt_pdb": wt_rel,
                "mut_pdb": mut_rel,
                "structure_status": status,
                "n_wt_ca": n_wt,
                "n_mut_ca": n_mut,
                "wt_has_mutation_site": wt_has_site,
                "mut_has_mutation_site": mut_has_site,
            }
            if args.include_residue_maps:
                row["wt_ca_residues"] = wt_map
                row["mut_ca_residues"] = mut_map
            rows.append(row)
        print(f"{split}: {n_ok}/{len(sample_ids)} complete WT+Mut pairs", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows)")

    if args.file_manifest is not None:
        args.file_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.file_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "sample_id", "role", "relpath", "n_ca"])
            writer.writeheader()
            writer.writerows(file_rows)
        print(f"Wrote {args.file_manifest} ({len(file_rows)} files)")


if __name__ == "__main__":
    main()
