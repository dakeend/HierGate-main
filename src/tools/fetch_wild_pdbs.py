#!/usr/bin/env python3
"""Download wild-type PDB files from RCSB for a HierGate mutation list.

Each mutation-list row starts with ``<PDB><chain>``. The experimental structure
used by HierGate is the RCSB entry for that 4-character PDB accession:

    https://files.rcsb.org/download/{PDB}.pdb

No privately collected coordinates are used. This script only fetches the public
asymmetric-unit PDB file; chain-filename matching is done by
``match_wild_chains.py``. Rosetta FastRelax is applied afterwards; this step
does not minimize, mutate, or energy-optimize coordinates.
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
USER_AGENT = "HierGate-structure-fetch/1.0 (https://github.com/dakeend/HierGate-main)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-l",
        "--mutant-list",
        required=True,
        type=Path,
        help="Mutation list with rows: PDBCHAIN POS WT MUT [DDG]",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for RCSB files named {PDB}.pdb, e.g. data/wild/S2648_ori",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between downloads (seconds)")
    return parser.parse_args()


def unique_pdb_ids(mutant_list: Path) -> list[str]:
    ids = []
    seen = set()
    with mutant_list.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            pdb_id = parts[0][:4].upper()
            if pdb_id not in seen:
                seen.add(pdb_id)
                ids.append(pdb_id)
    return ids


def download_pdb(pdb_id: str, dest: Path) -> None:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty response for {url}")
    dest.write_bytes(payload)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids = unique_pdb_ids(args.mutant_list)
    print(f"{len(pdb_ids)} unique PDB accessions from {args.mutant_list}")
    ok = skip = fail = 0
    for pdb_id in pdb_ids:
        dest = args.out_dir / f"{pdb_id}.pdb"
        if dest.exists() and not args.overwrite:
            print(f"[skip] {dest}")
            skip += 1
            continue
        try:
            download_pdb(pdb_id, dest)
            print(f"[ok] {pdb_id} -> {dest}")
            ok += 1
        except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
            print(f"[fail] {pdb_id}: {exc}")
            fail += 1
        time.sleep(args.sleep)
    print(f"done: downloaded={ok} skipped={skip} failed={fail}")


if __name__ == "__main__":
    main()
