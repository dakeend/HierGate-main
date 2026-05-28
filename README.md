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
|   +-- pdbs/                  # Optional relaxed protein structures, not bundled here
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
        |-- relax.py           # Rosetta FastRelax wrapper
        |-- hhblits.py         # HHblits profile generation helper
        +-- pssm_generator.py  # PSI-BLAST PSSM generation helper
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

Each sample is represented by a pair of graph files:

```text
<sample>_wt_rgcn.pkl
<sample>_mut_rgcn.pkl
```

The training split is based on S2648. The evaluation scripts also include commonly used benchmark datasets such as S350, S605, S1925, Myoglobin, P53, Ssym/test, S250, and S879.

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

- Rosetta FastRelax
- HH-suite3 and UniRef/UniClust database
- PSI-BLAST and Swiss-Prot database

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
  --split S2648 \
  --contact_threshold 5 \
  --knn 10 \
  --local_radius 12
```

Graph construction expects the corresponding relaxed PDB structures, HHM profiles, PSSM files, and amino-acid feature table to be available under the paths used by `src/utils/graph.py` and `src/utils/features.py`.

## Preprocessing Utilities

The repository includes helper scripts for generating structural and sequence-profile inputs:

### Rosetta Relaxation

```bash
python src/tools/relax.py \
  --mutant-list data/datasets/S2648_data.txt \
  --input-pdb-dir data/wild/S2648 \
  --rosetta-bin tools/rosetta/relax.static.linuxgccrelease \
  --output-dir data/pdbs/S2648
```

### HHblits Profiles

```bash
python src/tools/hhblits.py \
  --input-pdb-dir data/pdbs/S2648 \
  --hhsuite-db databases/UniRef30_2020_06/UniRef30_2020_06 \
  --output-dir data/hhm/S2648 \
  --cpu 8
```

### PSSM Profiles

```bash
python src/tools/pssm_generator.py
```

Before using the PSSM helper, configure `input_folder`, `output_folder`, and `blast_db_path` in `src/tools/pssm_generator.py`, or adapt the function `generate_pssm_from_folder` for your own pipeline.

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
- Regenerating graphs from raw structures requires external databases and software that are not bundled in the repository.
- The code currently contains some experiment-specific defaults, including task lists and GPU device selection.
- The project provides a lightweight `requirements.txt`; results may still depend on PyTorch, PyTorch Geometric, CUDA, and package versions.

## License

No license file is currently included in this repository. Add an explicit license before public release or redistribution.
