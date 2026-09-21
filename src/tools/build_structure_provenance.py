#!/usr/bin/env python3
"""Write a mutation-level RCSB provenance table for official HierGate splits.

The reviewer-requested structure source is the public PDB accession contained
in each sample name. This index lists, for every unique ``<PDB><chain>``:

- the RCSB download URL used by ``fetch_wild_pdbs.py``
- the local wild-type path expected by ``relax.py``
- how many mutations in the split use that structure
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

SPLIT_LISTS = {
    "s2648": "S2648_data.txt",
    "test": "s_sym.txt",
    "myoglobin": "myoglobin.txt",
    "p53": "p53.txt",
    "S250": "S250_data.txt",
    "S350": "S350_data.txt",
    "S605": "S605_data.txt",
    "S879": "S879_data.txt",
    "S1925": "S1925_data.txt",
    "S669": "S669_data.txt",
}

DISPLAY_NAME = {
    "s2648": "S2648",
    "test": "Ssym",
    "myoglobin": "Myoglobin",
    "p53": "p53",
    "S250": "S250",
    "S350": "S350",
    "S605": "S605",
    "S879": "S879",
    "S1925": "S1925",
    "S669": "S669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets"))
    parser.add_argument("--out", type=Path, default=Path("data/revision/structure_provenance.csv"))
    return parser.parse_args()


def load_proteins(path: Path) -> list[str]:
    proteins = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                proteins.append(parts[0])
    return proteins


def main() -> None:
    args = parse_args()
    rows = []
    for split, filename in SPLIT_LISTS.items():
        path = args.dataset_dir / filename
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        proteins = load_proteins(path)
        counts = Counter(proteins)
        for protein, n_mut in sorted(counts.items()):
            pdb_id = protein[:4].upper()
            chain = protein[4:]
            rows.append(
                {
                    "dataset": split,
                    "display_name": DISPLAY_NAME.get(split, split),
                    "protein": protein,
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "n_mutations": n_mut,
                    "rcsb_pdb_url": RCSB_PDB_URL.format(pdb_id=pdb_id),
                    "wild_local_path": f"data/wild/{split}/{protein}.pdb",
                    "structure_source": "RCSB PDB asymmetric unit (.pdb)",
                }
            )
        print(f"{split}: {len(counts)} unique structures, {len(proteins)} mutations")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} unique protein-chain entries)")


if __name__ == "__main__":
    main()
