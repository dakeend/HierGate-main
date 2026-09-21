#!/usr/bin/env python3
"""Keep wild-type PDBs and rename them to ``<PDB><chain>.pdb``.

Some source dumps use names such as ``1A43_A_wild_type.pdb``. HierGate's
FastRelax wrapper expects ``1A43A.pdb``. This script only copies and renames
wild-type files; it does not change coordinates.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def extract_pdb_chain_id(filename: str) -> str:
    base_name = filename.replace(".pdb", "")
    for pattern in (r"_wild.*", r"wild.*"):
        base_name = re.sub(pattern, "", base_name, flags=re.IGNORECASE)
    return base_name.replace("_", "")


def is_wild_type_pdb(filename: str) -> bool:
    return "wild" in filename.lower() and filename.lower().endswith(".pdb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir", required=True, type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    for file_path in sorted(args.input_dir.iterdir()):
        if not file_path.is_file() or not is_wild_type_pdb(file_path.name):
            continue
        new_filename = f"{extract_pdb_chain_id(file_path.name)}.pdb"
        target = args.output_dir / new_filename
        print(f"{file_path.name} -> {new_filename}")
        if not args.dry_run:
            shutil.copy2(file_path, target)
        processed += 1
    print(f"processed {processed} wild-type files")


if __name__ == "__main__":
    main()
