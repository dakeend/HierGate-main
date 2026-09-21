#!/usr/bin/env python3
"""Copy RCSB PDB files to HierGate ``<PDB><chain>.pdb`` names.

Rosetta FastRelax reads ``data/wild/<split>/<PDB><chain>.pdb``. RCSB files are
named ``<PDB>.pdb``. This script copies the fetched file once per chain that
appears in the mutation list. It does not edit ATOM records, strip heteroatoms,
renumber residues, or run any energy minimization.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-l", "--mutant-list", required=True, type=Path)
    parser.add_argument("-i", "--src-dir", required=True, type=Path, help="Directory of {PDB}.pdb files")
    parser.add_argument("-o", "--dest-dir", required=True, type=Path, help="Directory of {PDB}{chain}.pdb files")
    return parser.parse_args()


def needed_chain_ids(mutant_list: Path) -> set[str]:
    names = set()
    with mutant_list.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                names.add(parts[0])
    return names


def resolve_source(src_dir: Path, protein: str) -> Path | None:
    pdb_id = protein[:4]
    candidates = [
        src_dir / f"{protein}.pdb",
        src_dir / f"{pdb_id}.pdb",
        src_dir / f"{pdb_id.lower()}.pdb",
        src_dir / f"{pdb_id.upper()}.pdb",
        src_dir / f"{protein.lower()}.pdb",
        src_dir / f"{protein.upper()}.pdb",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def main() -> None:
    args = parse_args()
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    copied = skipped = missing = 0
    for protein in sorted(needed_chain_ids(args.mutant_list)):
        dest = args.dest_dir / f"{protein}.pdb"
        src = resolve_source(args.src_dir, protein)
        if src is None:
            print(f"[missing] {protein}")
            missing += 1
            continue
        if dest.exists():
            print(f"[skip] {dest}")
            skipped += 1
            continue
        shutil.copy2(src, dest)
        print(f"[copy] {src.name} -> {dest.name}")
        copied += 1
    print(f"done: copied={copied} skipped={skipped} missing={missing}")


if __name__ == "__main__":
    main()
