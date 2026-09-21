---
license: cc-by-4.0
task_categories:
- graph-ml
pretty_name: HierGate FastRelax PDB structures
---

# HierGate FastRelax PDB structures

Rosetta FastRelax wild-type and mutant PDB files used by [HierGate](https://github.com/dakeend/HierGate-main) for graph construction and feature extraction.

GitHub cannot host this release at full scale: the repository already contains preprocessed graphs, Git LFS has monthly bandwidth caps, and GitHub Releases cap individual files at 2 GB. This Hugging Face dataset is the canonical download location requested by Reviewer 1, comment 6.

## Contents

| File | Description |
| --- | --- |
| `structure_index.csv` | Mutation-level index: dataset, sample id, chain, mutation, WT/Mut relative paths |
| `pdb_file_manifest.csv` | Per-file listing of every packed `*_relaxed.pdb` |
| `archives/<split>_relaxed_pdb.tar.gz` | FastRelax structures for one official split |

Official splits: `s2648`, `test` (Ssym), `myoglobin`, `p53`, `S250`, `S350`, `S605`, `S879`, `S1925`, `S669`.

File naming:

```text
<PDB><chain>_relaxed.pdb
<PDB><chain>_<WT><pos><Mut>_relaxed.pdb
```

Example: `1AMQA_relaxed.pdb` and `1AMQA_C191Y_relaxed.pdb`.

## Download

```bash
pip install huggingface_hub
python src/tools/download_relaxed_pdbs.py --out-dir data/pdbs --extract
```

Or with the Hugging Face CLI:

```bash
hf download luwen486/HierGate-FastRelax-PDBs --repo-type dataset --local-dir data/pdbs
```
