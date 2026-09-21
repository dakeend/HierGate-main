#!/bin/bash
# Rebuild HierGate inputs from RCSB structures for one official split.
# External binaries and databases are not bundled with the GitHub repository.
set -euo pipefail

SPLIT=${1:?usage: run_feature_pipeline.sh <split> [mutant-list]}
LIST=${2:-data/datasets/${SPLIT}_data.txt}
if [ "$SPLIT" = "s2648" ]; then
  LIST=${2:-data/datasets/S2648_data.txt}
elif [ "$SPLIT" = "test" ]; then
  LIST=${2:-data/datasets/s_sym.txt}
fi

ROSETTA_BIN=${ROSETTA_BIN:-tools/rosetta/relax.static.linuxgccrelease}
HHSUITE_DB=${HHSUITE_DB:-databases/UniRef30_2020_06/UniRef30_2020_06}
BLAST_DB=${BLAST_DB:-databases/swissprot}

echo "Split=$SPLIT"
echo "Mutation list=$LIST"
echo "HHblits DB=$HHSUITE_DB   # UniRef30_2020_06"
echo "PSI-BLAST DB=$BLAST_DB   # NCBI BLAST Swiss-Prot snapshot 2025-07-06"

python src/tools/fetch_wild_pdbs.py --mutant-list "$LIST" --out-dir "data/wild/${SPLIT}_ori"
python src/tools/match_wild_chains.py --mutant-list "$LIST" --src-dir "data/wild/${SPLIT}_ori" --dest-dir "data/wild/${SPLIT}"
python src/tools/relax.py --mutant-list "$LIST" --input-pdb-dir "data/wild/${SPLIT}" --rosetta-bin "$ROSETTA_BIN" --output-dir "data/pdbs/${SPLIT}"
python src/tools/extract_sequences.py --input "data/pdbs/${SPLIT}" --output "data/xulie/${SPLIT}"
python src/tools/hhblits.py --input-pdb-dir "data/pdbs/${SPLIT}" --hhsuite-db "$HHSUITE_DB" --output-dir "data/hhm/${SPLIT}" --cpu "${HHBLITS_CPU:-8}"
python src/tools/pssm_generator.py --input-dir "data/xulie/${SPLIT}" --output-dir "data/pssm/${SPLIT}" --blast-db "$BLAST_DB"
python gen_graph.py --data_path "$LIST" --out_dir data/Rgraph_80 --split "$SPLIT"
