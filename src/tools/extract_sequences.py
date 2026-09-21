#!/usr/bin/env python3
"""Extract one-letter sequences from FastRelax PDB files for PSI-BLAST.

HMM profiles are generated directly from relaxed PDBs by ``hhblits.py``.
PSSM profiles are generated from these sequence text files by
``pssm_generator.py``. Sequences are read from CA atoms; non-standard residues
are mapped when known (MSE->M) and otherwise written as X.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

AA_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


def extract_sequence_from_pdb(pdb_file: Path) -> dict[str, str]:
    sequence: dict[str, OrderedDict[int, str]] = {}
    with pdb_file.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") and " CA " in line):
                continue
            res_name = line[17:20].strip()
            res_num = int(line[22:26].strip())
            chain_id = line[21].strip()
            sequence.setdefault(chain_id, OrderedDict())
            if res_num not in sequence[chain_id]:
                sequence[chain_id][res_num] = res_name
    result = {}
    for chain_id, residues in sequence.items():
        result[chain_id] = "".join(AA_MAP.get(residues[res_num], "X") for res_num in sorted(residues))
    return result


def stem_without_relaxed(name: str) -> str:
    if name.endswith("_relaxed"):
        return name[: -len("_relaxed")]
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=Path("data/pdbs/S2648"))
    parser.add_argument("-o", "--output", type=Path, default=Path("data/xulie/S2648"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pdb_count = sequence_count = 0
    for pdb_path in args.input.rglob("*.pdb"):
        pdb_count += 1
        sequences = extract_sequence_from_pdb(pdb_path)
        pdb_name = pdb_path.stem
        if not sequences:
            print(f"[empty] {pdb_path}")
            continue
        for chain_id, sequence in sequences.items():
            if not sequence:
                continue
            if len(sequences) > 1:
                out_file = args.output / f"{pdb_name}_{chain_id}.txt"
            else:
                out_file = args.output / f"{stem_without_relaxed(pdb_name)}.txt"
            out_file.write_text(sequence, encoding="utf-8")
            sequence_count += 1
            print(f"{pdb_path} -> {out_file.name}")
    print(f"processed {pdb_count} PDBs, wrote {sequence_count} sequences")


if __name__ == "__main__":
    main()
