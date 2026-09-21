#!/usr/bin/env python3
"""Download FastRelax WT/Mut PDB archives released with HierGate.

The structures are hosted on Hugging Face because they are too large for GitHub.
Default repository: https://huggingface.co/datasets/luwen486/HierGate-FastRelax-PDBs
"""

from __future__ import annotations

import argparse
from pathlib import Path


HF_REPO = "luwen486/HierGate-FastRelax-PDBs"
SPLITS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=HF_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--out-dir", type=Path, default=Path("data/pdbs"))
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--index-only", action="store_true", help="Download structure_index.csv only")
    parser.add_argument("--extract", action="store_true", help="Extract tar.gz archives after download")
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index_path = hf_hub_download(
        repo_id=args.repo,
        repo_type="dataset",
        filename="structure_index.csv",
        local_dir=str(args.out_dir),
    )
    print(f"Index: {index_path}")
    if args.index_only:
        return

    allow = [f"archives/{split}_relaxed_pdb.tar.gz" for split in args.splits]
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        allow_patterns=allow + ["structure_index.csv", "pdb_file_manifest.csv", "README.md"],
        local_dir=str(args.out_dir),
    )
    if args.extract:
        import tarfile

        for split in args.splits:
            archive = args.out_dir / "archives" / f"{split}_relaxed_pdb.tar.gz"
            if not archive.exists():
                print(f"[skip extract] {archive}")
                continue
            print(f"Extracting {archive}")
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(args.out_dir)


if __name__ == "__main__":
    main()
