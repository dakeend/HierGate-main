"""Generate HHM profiles with HHblits.

HierGate HMM search database
----------------------------
- Name / version: UniRef30_2020_06
- Files: UniRef30_2020_06_*.ffdata / UniRef30_2020_06_*.ffindex
- Download:
  https://wwwuser.gwdguser.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
- Original data-file date: 2020-10-05
- Command used here: hhblits -d UniRef30_2020_06 -n 3

Sequences are read from FastRelax PDB files. This script does not change
coordinates.
"""

import os
import argparse
import warnings
from tempfile import NamedTemporaryFile

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


def pdb2seq(pdb_dir):
    ppb = PPBuilder()
    records = []

    # 遍历主目录下的所有子文件夹
    for subdir in os.listdir(pdb_dir):
        subdir_path = os.path.join(pdb_dir, subdir)

        # 确保是文件夹
        if os.path.isdir(subdir_path):
            print(f"Processing folder: {subdir}")

            # 遍历子文件夹中的所有文件
            for pdb_file in os.listdir(subdir_path):
                if pdb_file.endswith('.pdb'):
                    pdb_path = os.path.join(subdir_path, pdb_file)
                    print(f"  Found PDB file: {pdb_file}")

                    try:
                        structure = PDBParser(QUIET=True).get_structure('pdb', pdb_path)

                        pdb = os.path.splitext(os.path.basename(pdb_path))[0]
                        pdb = pdb.replace('_relaxed', '')

                        chain_name = pdb[4]  # 第5个字符（索引为4）


                        chain = structure[0][chain_name]

                        pp = ppb.build_peptides(chain)

                        sequence = ''.join([str(p.get_sequence()) for p in pp])
                        record = SeqRecord(Seq(sequence), id=pdb, description='')
                        records.append(record)

                        print(f"    Successfully processed: {pdb} (chain {chain_name})")

                    except Exception as e:
                        print(f"    Error processing {pdb_file}: {str(e)}")
                        continue

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Use hhblits to generate .hhm files against UniRef30_2020_06")
    parser.add_argument('-i', '--input-pdb-dir', type=str, dest='input_pdb_dir',
                        default='data/pdbs/S1131',
                        help='The directory storing the PDB files.')
    parser.add_argument('-db', '--hhsuite-db', type=str, dest="hhsuite_db",
                        default='databases/UniRef30_2020_06/UniRef30_2020_06',
                        help='HHsuite prefix for UniRef30_2020_06 '
                             '(https://wwwuser.gwdguser.de/~compbiol/uniclust/2020_06/'
                             'UniRef30_2020_06_hhsuite.tar.gz; data files dated 2020-10-05).')
    parser.add_argument('-o', '--output-dir', type=str, dest="output_dir",
                        default='data/hhm/S1131',
                        help='The directory to store all output data.')
    parser.add_argument('--cpu', type=str, default='1',
                        help='number of CPUs to use (default: 4)')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Scanning directory: {args.input_pdb_dir}")
    records = pdb2seq(args.input_pdb_dir)

    print(f"\nFound {len(records)} PDB files to process")
    print(f"Output directory: {args.output_dir}")

    for i, record in enumerate(records, 1):
        print(f"\nProcessing {i}/{len(records)}: {record.id}")

        f = NamedTemporaryFile(prefix='tmp', suffix='.fasta', delete=False)
        SeqIO.write([record], f.name, "fasta")

        output_hhm = os.path.join(args.output_dir, record.id + ".hhm")

        hhblits_cmd = ' '.join(['hhblits', '-i', f.name, '-o', '/dev/null',
                                '-ohhm', output_hhm,
                                '-d', args.hhsuite_db, '-n 3', '-cpu', args.cpu])

        print(f"Command: {hhblits_cmd}")
        result = os.system(hhblits_cmd)

        # 清理临时文件
        os.unlink(f.name)

        if result == 0:
            print(f"Successfully generated: {output_hhm}")
        else:
            print(f"Error running hhblits for {record.id}")

    print(f"\nProcessing complete! Generated files in: {args.output_dir}")


if __name__ == "__main__":
    main()
