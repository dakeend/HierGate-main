import argparse
import os
import warnings

from src.utils.features import load_aa_features
from src.utils.graph import make_rgcn_graph



def main():
    parser = argparse.ArgumentParser(description='Generate graphs for GNN model')
    parser.add_argument('--feature_path', type=str, default='data/features.txt',
                        help='path to file saving sequence encoding features')
    parser.add_argument('--data_path', type=str, default='data/datasets/S879_data.txt',
                        help='path to file recording mutations and ddGs')
    parser.add_argument('--out_dir', type=str, default='data/Rgraph_80',
                        help='directory to save the output graphs')
    parser.add_argument('--split', type=str, default="S879",
                        help='split for different dataset (train, test, p53, myoglobin)')
    parser.add_argument('--contact_threshold', type=float, default=5,
                        help='threshold for contact edge between residues (defalut: 5)')
    parser.add_argument('--knn', type=float, default=10,
                        help='threshold for contact edge between residues (defalut: 5)')
    parser.add_argument('--local_radius', type=float, default=12,
                        help='maximum distance from the mutation postion (default: 12)')
    parser.add_argument('--edge_feature_mode', type=str, default="mdtraj",
                        choices=["builtin", "mdtraj", "none"],
                        help='edge feature mode: builtin (ProteinEdgeFeatureGenerator), mdtraj (MDTraj), or none (basic weights)')
    parser.add_argument('--use_enhanced_edge_features', action='store_true', default=True,
                        help='whether to use enhanced edge features')
    parser.add_argument('--no_enhanced_edge_features', dest='use_enhanced_edge_features',
                        action='store_false',
                        help='disable enhanced edge features')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    if not os.path.exists(os.path.join(args.out_dir, args.split)):
        os.makedirs(os.path.join(args.out_dir, args.split))

    aa_features = load_aa_features(args.feature_path)
    for i, record in enumerate(open(args.data_path)):
        print(f"Processing {i + 1}th record: {record.strip()}")

        try:

            make_rgcn_graph(
                record=record,
                aa_features=aa_features,
                is_wt=True,
                split=args.split,
                out_dir=args.out_dir,
                contact_threshold=args.contact_threshold,  # 空间距离阈值
                k=args.knn,  # KNN邻居数
                seq_window=2,  # 序列窗口
                local_radius=args.local_radius,
            )

            make_rgcn_graph(
                record=record,
                aa_features=aa_features,
                is_wt=False,
                split=args.split,
                out_dir=args.out_dir,
                contact_threshold=args.contact_threshold,  # 空间距离阈值
                k=args.knn,  # KNN邻居数
                seq_window=2,  # 序列窗口
                local_radius=args.local_radius,
            )
        except Exception as e:
            print(f"Error processing record {record.strip()}: {e}")
            continue

    print("Graph generation completed!")


if __name__ == "__main__":
    main()
