# HierGate: Hierarchically Gated Multiplex Heterogeneous Graph Neural Network for Protein Stability Prediction

HierGate is a structure-based graph neural network framework for predicting mutation-induced protein thermal stability changes. Given a wild-type protein structure and its corresponding mutant structure, the model predicts the experimental stability change, commonly denoted as DeltaDeltaG.

This repository contains the implementation, preprocessed graph datasets, and trained five-fold model checkpoints used for evaluating HierGate on protein stability prediction benchmarks.

## Overview

Single-residue mutations may perturb protein three-dimensional structure and thermal stability, which is important for protein engineering, disease mechanism analysis, and variant interpretation. Existing structure-based models often rely on a single contact graph and may underuse the heterogeneous relations among residues. They may also lack explicit mechanisms for filtering noisy structural context and for strengthening weak mutation-level supervision.

HierGate addresses these issues through three components:

- **Multiplex heterogeneous residue graphs**: local residue graphs are constructed around the mutation site and encode complementary relation types, including sequential proximity, spatial contact, and K-nearest-neighbor relations.
- **Hierarchical gating**: relation-level and feature-level gating mechanisms are used to filter contextual noise during heterogeneous message aggregation.
- **Self-distillation**: deep-layer representations provide auxiliary supervision to shallow layers, helping amplify sparse mutation signals in the DeltaDeltaG prediction task.

The model follows a paired wild-type/mutant graph formulation. For each mutation, two graphs are loaded: one for the wild-type structure and one for the mutant structure. The model learns from both direct and reverse mutation pairs and reports correlation and RMSE-based evaluation metrics.

## Repository Structure

```text
.
|-- main.py                    # Training and cross-validation entry point
|-- predict.py                 # Prediction/evaluation entry point using trained checkpoints
|-- gen_graph.py               # Graph generation from prepared structural features
|-- MHGGN_main.tex             # Manuscript draft describing the HierGate method
|-- README_thermoGNN.md        # Reference README from ThermoGNN
|-- data/
|   |-- Rgraph_80/             # Preprocessed graph files
|   |-- R3_names/              # Split-specific sample name lists
|   |-- datasets/              # Mutation list files and source tabular data
|   |-- hhm/                   # HHblits profile files
|   |-- pssm/                  # PSSM files
|   |-- pdbs/                  # Optional relaxed protein structures, not bundled here
|   +-- revision/              # Structure provenance, FastRelax index, case-study tables
|-- run/
|   +-- best/
|       |-- config.json        # Configuration for the provided checkpoints
|       |-- model_1.pkl        # Fold-1 checkpoint
|       |-- model_2.pkl        # Fold-2 checkpoint
|       |-- model_3.pkl        # Fold-3 checkpoint
|       |-- model_4.pkl        # Fold-4 checkpoint
|       +-- model_5.pkl        # Fold-5 checkpoint
+-- src/
    |-- dataset.py             # PairData construction and dataset loading
    |-- model.py               # GraphGNN and paired graph prediction model
    |-- training.py            # Training, evaluation, metrics, and early stopping
    |-- loss.py                # Loss functions and curriculum-related losses
    |-- rgcn_gate.py           # Relation-aware gated RGCN layer
    |-- utils/
    |   |-- graph.py           # Multiplex heterogeneous graph construction
    |   |-- features.py        # HHM, PSSM, Rosetta, and amino-acid feature readers
    |   |-- fds.py             # Feature distribution smoothing utilities
    |   |-- weights.py         # Label distribution smoothing weights
    |   +-- utils.py           # Plotting and reporting utilities
    +-- tools/
        |-- fetch_wild_pdbs.py # Download RCSB PDB files from mutation lists
        |-- match_wild_chains.py
        |-- filter_wild_pdb.py
        |-- relax.py           # Rosetta 2021.16.61629 FastRelax wrapper
        |-- extract_sequences.py
        |-- hhblits.py         # HHblits / UniRef30_2020_06
        |-- pssm_generator.py  # PSI-BLAST / NCBI BLAST Swiss-Prot
        |-- build_structure_provenance.py
        |-- build_structure_index.py
        |-- download_relaxed_pdbs.py
        +-- run_feature_pipeline.sh
```

## Data

The implementation expects preprocessed graph files under `data/Rgraph_80/<split>/` and split-specific sample names under `data/R3_names/<split>_names.txt`.

The repository currently includes graph splits for common stability benchmarks, including:

- `s2648`
- `test`
- `myoglobin`
- `p53`
- `S250`
- `S350`
- `S605`
- `S879`
- `S1925`
- `S669`

Each sample is represented by a pair of graph files:

```text
<sample>_wt_rgcn.pkl
<sample>_mut_rgcn.pkl
```

The training split is based on S2648. The evaluation scripts also include commonly used benchmark datasets such as S350, S605, S1925, Myoglobin, P53, Ssym/test, S250, S879, and S669.

### Case-study predicted ΔΔG tables (p53 and BRCA1)

The complete predicted ΔΔG tables for every analyzed single-site substitution in the p53 and BRCA1 case studies are in this repository, not only as heatmap figures:

- p53 (PDB 2OCJ, chain A; residues 164–289; 2,394 predictions): [`data/revision/case_studies/Table_S1_p53_full_sequence_predictions.csv`](https://github.com/dakeend/HierGate-main/blob/main/data/revision/case_studies/Table_S1_p53_full_sequence_predictions.csv)
- BRCA1 (PDB 1T15, chain A; residues 1765–1859; 1,805 predictions): [`data/revision/case_studies/Table_S2_BRCA1_full_sequence_predictions.csv`](https://github.com/dakeend/HierGate-main/blob/main/data/revision/case_studies/Table_S2_BRCA1_full_sequence_predictions.csv)

Each row contains residue position, wild-type amino acid, mutant amino acid, and predicted ΔΔG (kcal/mol). These scans are the values used for the case-study figures; they are not replaced by the 42-mutation p53 external test set.

## Original structures, preprocessing, and profile databases

Rebuilding graphs from raw proteins requires three pieces of information: where each experimental structure came from, what (if anything) was done to it before Rosetta, and which HMM/PSSM libraries were searched. The scripts under `src/tools/` implement that pipeline.

### Structure source

Every mutation-list row in `data/datasets/` starts with `<PDB><chain>`. The experimental wild-type coordinates are the public RCSB PDB entry for that 4-character accession:

```text
https://files.rcsb.org/download/{PDB}.pdb
```

No privately collected structures are used. The unique protein-chain index, with RCSB URLs and the local path expected by FastRelax, is:

```text
data/revision/structure_provenance.csv
```

Rebuild the index:

```bash
python src/tools/build_structure_provenance.py \
  --dataset-dir data/datasets \
  --out data/revision/structure_provenance.csv
```

Download and rename wild-type files for one split:

```bash
python src/tools/fetch_wild_pdbs.py \
  --mutant-list data/datasets/S2648_data.txt \
  --out-dir data/wild/s2648_ori

python src/tools/match_wild_chains.py \
  --mutant-list data/datasets/S2648_data.txt \
  --src-dir data/wild/s2648_ori \
  --dest-dir data/wild/s2648
```

`match_wild_chains.py` copies `{PDB}.pdb` to `{PDB}{chain}.pdb`. If a source dump already uses names such as `1A43_A_wild_type.pdb`, use `src/tools/filter_wild_pdb.py` instead. Both scripts only copy and rename files.

### Preprocessing before Rosetta

Before FastRelax, HierGate does **not** run extra coordinate minimization, mutation, or energy optimization. The only structure-side steps are:

1. Fetch the RCSB PDB file for the accession in the mutation list.
2. Copy/rename it so that FastRelax reads `data/wild/<split>/<PDB><chain>.pdb`.
3. Match the chain character in the filename to the mutation-list chain ID.

ATOM records are not stripped, renumbered, or rebuilt at this stage. The first coordinate refinement is Rosetta FastRelax (Rosetta **2021.16.61629**):

- wild-type: `-in:file:fullatom -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -detect_disulf false`
- mutant: the same flags, plus `-relax:respect_resfile -packing:resfile`

```bash
python src/tools/relax.py \
  --mutant-list data/datasets/S2648_data.txt \
  --input-pdb-dir data/wild/s2648 \
  --rosetta-bin tools/rosetta/relax.static.linuxgccrelease \
  --output-dir data/pdbs/s2648
```

Relaxed files are stored as `<PDB><chain>_relaxed.pdb` and `<PDB><chain>_<WT><pos><Mut>_relaxed.pdb`. The FastRelax WT/Mut PDBs used for graph construction are too large for GitHub, so they are released on Hugging Face with a mutation-level index:

- Dataset: https://huggingface.co/datasets/luwen486/HierGate-FastRelax-PDBs
- Mutation-level index: `structure_index.csv` in that dataset, also copied to `data/revision/structure_release/structure_index.csv`

Download and extract:

```bash
python src/tools/download_relaxed_pdbs.py --out-dir data/pdbs --extract
```

Or:

```bash
hf download luwen486/HierGate-FastRelax-PDBs --repo-type dataset --local-dir data/pdbs
```

### HMM database (UniRef30_2020_06)

HHM profiles are generated from relaxed PDB sequences with HHblits, **3 iterations**, against **UniRef30_2020_06**.

| Item | Value |
| --- | --- |
| Database | UniRef30_2020_06 |
| Files | `UniRef30_2020_06_*.ffdata` / `UniRef30_2020_06_*.ffindex` |
| Download | https://wwwuser.gwdguser.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz |
| Original data-file date | 2020-10-05 |
| Command | `hhblits -d UniRef30_2020_06 -n 3` |
| Script | `src/tools/hhblits.py` |
| Output | `data/hhm/<split>/<suffix>.hhm` |

```bash
python src/tools/hhblits.py \
  --input-pdb-dir data/pdbs/s2648 \
  --hhsuite-db databases/UniRef30_2020_06/UniRef30_2020_06 \
  --output-dir data/hhm/s2648 \
  --cpu 8
```

### PSSM database (NCBI BLAST Swiss-Prot, 2025-07-06 snapshot)

PSSM profiles are generated with PSI-BLAST from sequences extracted by `src/tools/extract_sequences.py`. The search library is an **NCBI BLAST-format Swiss-Prot** database. It is **not** labeled with a UniProt `2025_xx` release tag.

| Item | Value |
| --- | --- |
| Local path used in the original run | `/media/ST-18T/nianwen/pssm_project/databases/swissprot` |
| `blastdbcmd -info` / `swissprot.pjs` name | Non-redundant UniProtKB/SwissProt sequences |
| Date / last-updated | 2025-07-06 04:44 |
| Sequences | 485,565 |
| BLASTDB Version | 5 (NCBI BLAST *format* version, not a UniProt release) |
| `.phr` contents | protein entry names only; no UniProt `2025_xx` string; no README in that directory |
| PSI-BLAST flags | `-evalue 0.001 -num_iterations 3 -out_ascii_pssm` |
| Script | `src/tools/pssm_generator.py` |
| Output | `data/pssm/<split>/<suffix>.pssm` |

```bash
python src/tools/extract_sequences.py \
  --input data/pdbs/s2648 \
  --output data/xulie/s2648

python src/tools/pssm_generator.py \
  --input-dir data/xulie/s2648 \
  --output-dir data/pssm/s2648 \
  --blast-db databases/swissprot
```

A one-split wrapper that runs fetch → chain match → FastRelax → sequence → HMM → PSSM → graph is:

```bash
bash src/tools/run_feature_pipeline.sh s2648 data/datasets/S2648_data.txt
```

Set `ROSETTA_BIN`, `HHSUITE_DB`, and `BLAST_DB` if the binaries/databases are not in the default paths.

## Installation

Python package dependencies are listed in `requirements.txt`. The following packages are required by the code:

- Python 3.9 or later
- PyTorch
- PyTorch Geometric
- torchmetrics
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib
- NetworkX
- Biopython

An example conda-based setup is:

```bash
conda create -n hiergate python=3.10
conda activate hiergate

# Install PyTorch according to your CUDA version from the official PyTorch instructions.
# Example CPU-only command:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install PyTorch Geometric according to your PyTorch/CUDA version.
conda install pyg -c pyg

conda install numpy scipy scikit-learn pandas matplotlib networkx -c conda-forge
conda install biopython -c bioconda
pip install torchmetrics
```

Alternatively, after installing a platform-compatible PyTorch build, install the remaining Python dependencies with:

```bash
pip install -r requirements.txt
```

For graph generation from raw structures, additional external tools and databases are required:

- Rosetta 2021.16.61629 FastRelax
- HH-suite3 with **UniRef30_2020_06**
  ([HHsuite tarball](https://wwwuser.gwdguser.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz); data files dated 2020-10-05)
- PSI-BLAST and an NCBI BLAST-format **Swiss-Prot** library. The snapshot used here is `Non-redundant UniProtKB/SwissProt sequences`, last-updated **2025-07-06 04:44**, 485,565 sequences, BLASTDB version 5. That version number is the BLAST database format, not a UniProt `2025_xx` release.

These external resources are not bundled with this repository.

## Quick Start: Prediction with Provided Checkpoints

The repository includes five trained checkpoints in `run/best/`, named from `run/best/model_1.pkl` to `run/best/model_5.pkl`. To evaluate the provided model on the default benchmark task list:

```bash
python predict.py --graph-dir data/Rgraph_80 --weight-dir run/best --split 5
```

The prediction script loads `model_1.pkl` through `model_5.pkl`, averages fold predictions, and reports PCC/RMSE metrics for the configured datasets. 

## Training

To train HierGate on the preprocessed S2648 graph split:

```bash
python main.py --graph-dir data/Rgraph_80 --logging-dir .
```

By default, the training script:

- loads `s2648` as the training split;
- constructs direct and reverse mutation pairs;
- performs five-fold cross-validation;
- saves fold checkpoints as `model_1.pkl` through `model_5.pkl`;
- evaluates case-study benchmark splits after training.

Important runtime note: the current scripts explicitly select CUDA devices inside `main.py` and `predict.py`. If running on CPU or a single-GPU machine, review the device-selection lines before execution.

## Graph Generation

If raw structural features are available, graphs can be regenerated with:

```bash
python gen_graph.py \
  --feature_path data/features.txt \
  --data_path data/datasets/S2648_data.txt \
  --out_dir data/Rgraph_80 \
  --split s2648 \
  --contact_threshold 5 \
  --knn 10 \
  --local_radius 12
```

Graph construction reads FastRelax PDBs from `data/pdbs/<split>/<PDB><chain>/<suffix>_relaxed.pdb`, HHM files from `data/hhm/<split>/`, PSSM files from `data/pssm/<split>/`, and the amino-acid feature table. See **Original structures, preprocessing, and profile databases** above for how those files are produced.

## Method Summary

HierGate formulates protein stability prediction as a paired-graph regression problem. Given a wild-type protein and a mutant protein, local residue environments are extracted around the mutation site and encoded as multiplex heterogeneous residue graphs.

The graph contains multiple residue relation families:

1. **Sequential relations**, which connect adjacent residues along the amino-acid sequence.
2. **Spatial contact relations**, which connect residues according to C-alpha distance thresholds.
3. **K-nearest-neighbor relations**, which model relative spatial proximity in the three-dimensional structure.

Residue nodes are described using structural, energetic, evolutionary, and physicochemical descriptors. The manuscript draft describes a descriptor design combining amino-acid encoding, Rosetta energy terms, HHblits-derived evolutionary profiles, and PSI-BLAST PSSM features.

The neural architecture uses a multiplex heterogeneous graph neural network to propagate information over relation-aware graphs. Hierarchical gating is used to regulate information flow both across relation types and within feature channels, reducing the influence of irrelevant context around the mutation site. A self-distillation strategy further supervises shallow representations using deeper task-aware predictions, improving learning when mutation signals are sparse and weak.

## Reproducibility Notes

- The provided checkpoints are stored in `run/best/`.
- The configuration associated with these checkpoints is stored in `run/best/config.json`.
- Preprocessed graph files are already available for the main benchmark splits.
- Wild-type experimental structures are RCSB PDB files identified by the accession in each mutation list (`data/revision/structure_provenance.csv`).
- Before FastRelax, structures are only copied and renamed to `<PDB><chain>.pdb`; they are not extra-minimized.
- HMM profiles used **UniRef30_2020_06** (HHsuite tarball above; data files dated 2020-10-05).
- PSSM profiles used an NCBI BLAST Swiss-Prot snapshot dated **2025-07-06 04:44** (485,565 sequences, BLASTDB version 5). This cannot be written as a UniProt `2025_xx` release.
- Regenerating graphs from raw structures requires Rosetta, HHblits, PSI-BLAST, and those two databases.
- The FastRelax WT/Mut PDBs used in this work are at https://huggingface.co/datasets/luwen486/HierGate-FastRelax-PDBs.
- Predicted ΔΔG tables for the p53 and BRCA1 case-study scans are at `data/revision/case_studies/`.
- The code currently contains some experiment-specific defaults, including task lists and GPU device selection.
- The project provides a lightweight `requirements.txt`; results may still depend on PyTorch, PyTorch Geometric, CUDA, and package versions.
