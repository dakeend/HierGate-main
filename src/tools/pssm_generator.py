#!/usr/bin/env python3
"""Generate ASCII PSSM files with PSI-BLAST.

HierGate PSSM search database
-----------------------------
The profiles used in this work were built with PSI-BLAST 2.12.0+ (NCBI BLAST+
2.12.0) against Swiss-Prot release 2025_03, stored as an NCBI BLAST database.

- Program: PSI-BLAST 2.12.0+ (package blast 2.12.0, build Mar 8 2022)
- Swiss-Prot release: 2025_03
- Local path used in the original run:
  ``/media/ST-18T/nianwen/pssm_project/databases/swissprot``
- ``blastdbcmd -info`` / ``swissprot.pjs`` library name:
  Non-redundant UniProtKB/SwissProt sequences
- Date / last-updated: 2025-07-06 04:44
- Sequences: 485,565
- Residues: 184,945,355
- BLASTDB Version: 5  (NCBI BLAST database *format* version, not the
  Swiss-Prot release number)

PSI-BLAST parameters: ``-evalue 0.001 -num_iterations 3 -out_ascii_pssm``.
"""

from __future__ import annotations

import argparse
import glob
import os

PSIBLAST_VERSION = "2.12.0+"
SWISSPROT_RELEASE = "2025_03"
SWISSPROT_BLASTDB_NAME = "Non-redundant UniProtKB/SwissProt sequences"
SWISSPROT_LAST_UPDATED = "2025-07-06 04:44"
SWISSPROT_N_SEQUENCES = 485565
SWISSPROT_N_RESIDUES = 184945355
SWISSPROT_BLASTDB_VERSION = 5


def extract_base_name(filename):
    """
    从文件名提取基础名称
    1a5eA.txt -> 1a5e
    1a5eA_L37S.txt -> 1a5e
    """
    base = os.path.splitext(filename)[0]
    if "_" in base:
        base = base.split("_")[0]
    if len(base) > 4:
        base = base[:-1]
    return base


def generate_pssm_from_folder(input_folder, output_folder, blast_db_path, evalue="0.001", num_iterations=3):
    """
    从文件夹中读取所有txt文件，生成PSSM特征

    Args:
        input_folder: 包含txt序列文件的文件夹路径
        output_folder: PSSM文件输出文件夹路径
        blast_db_path: BLAST数据库路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    if not txt_files:
        print("未找到任何txt文件")
        return

    print(f"找到 {len(txt_files)} 个txt文件")
    print(
        "PSI-BLAST "
        f"{PSIBLAST_VERSION} DB: "
        f"{blast_db_path} | Swiss-Prot {SWISSPROT_RELEASE} | {SWISSPROT_BLASTDB_NAME} | "
        f"last-updated {SWISSPROT_LAST_UPDATED} | "
        f"n={SWISSPROT_N_SEQUENCES} residues={SWISSPROT_N_RESIDUES} | "
        f"BLASTDB v{SWISSPROT_BLASTDB_VERSION}"
    )
    temp_fasta = os.path.join(input_folder, "Temporary.fasta")

    for txt_file in txt_files:
        try:
            filename = os.path.basename(txt_file)
            print(f"正在处理: {filename}")
            with open(txt_file, "r") as f:
                sequence = f.read().strip()
            base_name = extract_base_name(filename)
            with open(temp_fasta, "w") as f:
                f.write(f">{base_name}\n")
                f.write(sequence)
            output_pssm = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.pssm")
            psiblast_cmd = (
                f'psiblast -query "{temp_fasta}" '
                f'-db "{blast_db_path}" '
                f"-evalue {evalue} -num_iterations {num_iterations} "
                f'-out_ascii_pssm "{output_pssm}"'
            )
            result = os.system(psiblast_cmd)
            if result == 0:
                print(f"  ✓ {filename} -> {os.path.basename(output_pssm)}")
            else:
                print(f"  ✗ {filename} 处理失败")
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
    print("所有PSSM矩阵构建完成")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir", default="data/xulie/S2648")
    parser.add_argument("-o", "--output-dir", default="data/pssm/S2648")
    parser.add_argument(
        "-d",
        "--blast-db",
        default="databases/swissprot",
        help=(
            "NCBI BLAST-format Swiss-Prot prefix. HierGate used PSI-BLAST "
            f"{PSIBLAST_VERSION} against Swiss-Prot release {SWISSPROT_RELEASE} "
            f"({SWISSPROT_BLASTDB_NAME}), last-updated {SWISSPROT_LAST_UPDATED}, "
            f"{SWISSPROT_N_SEQUENCES} sequences, {SWISSPROT_N_RESIDUES} residues, "
            f"BLASTDB version {SWISSPROT_BLASTDB_VERSION} "
            "(format version, not the Swiss-Prot release number)."
        ),
    )
    parser.add_argument("--evalue", default="0.001")
    parser.add_argument("--num-iterations", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pssm_from_folder(
        args.input_dir,
        args.output_dir,
        args.blast_db,
        evalue=args.evalue,
        num_iterations=args.num_iterations,
    )
