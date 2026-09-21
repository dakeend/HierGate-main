# Case-study predicted ΔΔG tables

Complete single-site substitution scans used in the p53 and BRCA1 case studies (Reviewer 3, major comment 3). These are the predictions behind the case-study heatmaps, not the 42-mutation p53 external test set.

| Table | Protein | PDB / chain | Scan range | N | File |
| --- | --- | --- | --- | --- | --- |
| S1 | p53 | 2OCJ / A | 164–289 (126 sites × 19 substitutions) | 2,394 | `Table_S1_p53_full_sequence_predictions.csv` |
| S2 | BRCA1 | 1T15 / A | 1765–1859 (95 sites × 19 substitutions) | 1,805 | `Table_S2_BRCA1_full_sequence_predictions.csv` |

Each row includes residue position, wild-type amino acid, mutant amino acid, mutation string, and predicted ΔΔG (kcal/mol). Sign convention: ΔΔG = ΔG_WT − ΔG_Mut; positive values indicate stabilization.

Direct links:

- https://github.com/dakeend/HierGate-main/blob/main/data/revision/case_studies/Table_S1_p53_full_sequence_predictions.csv
- https://github.com/dakeend/HierGate-main/blob/main/data/revision/case_studies/Table_S2_BRCA1_full_sequence_predictions.csv
