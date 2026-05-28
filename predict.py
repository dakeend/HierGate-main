import argparse
import os
import torch
import logging
from torch_geometric.loader import DataLoader

# 修改导入以匹配main文件
from src.dataset import load_dataset
from src.model import GraphGNN
from src.training import evaluate, metrics

from src.utils.utils import plot_results_enhanced_v2
import torch
import numpy as np
from scipy.stats import pearsonr


def compute_antisymmetry(pred_dir, pred_rev, y_dir=None):
    """
    计算 ΔΔG 预测的反对称性指标：
      1. R_FR : corr(pred_dir, -pred_rev)
      2. delta: mean(pred_dir + pred_rev)

    参数:
      pred_dir: 正向预测 (tensor 或 numpy)
      pred_rev: 反向预测
      y_dir   : 正向真实值（可选）

    返回:
      R_FR, delta
    """

    # 转换到 numpy
    if isinstance(pred_dir, torch.Tensor):
        pred_dir = pred_dir.detach().cpu().numpy()
    if isinstance(pred_rev, torch.Tensor):
        pred_rev = pred_rev.detach().cpu().numpy()
    if y_dir is not None and isinstance(y_dir, torch.Tensor):
        y_dir = y_dir.detach().cpu().numpy()

    # ------- 1) 反向相关系数 R_FR -------
    try:
        R_FR, _ = pearsonr(pred_dir, pred_rev)
    except Exception:
        R_FR = float("nan")

    # ------- 2) 偏移量 delta -------
    delta = np.mean(pred_dir + pred_rev)

    # ------- 如果提供真实值，可进一步验证 --------

    return R_FR, delta

def predict_and_save(args, model, task, graph_dir, weight_dir, fold=5,color='#274753'):
    """
    Load trained models, predict on test datasets, and save predictions to txt files
    """
    logging.info(f"Task: {task}")
    
    # 修改数据集加载以匹配main文件的格式
    test_data_list = load_dataset(graph_dir, task)
    test_direct_dataset, test_reverse_dataset = test_data_list[::2], test_data_list[1::2]
    
    test_loader = DataLoader(
        test_direct_dataset+test_reverse_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)
    test_direct_loader = DataLoader(
        test_direct_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)
    test_reverse_loader = DataLoader(
        test_reverse_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)

    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    torch.cuda.set_device(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)

    # Store predictions from each fold
    total_pred_dir = []
    total_pred_rev = []
    total_pred_all = []
    # Get ground truth values from first batch
    with torch.no_grad():
        for data in test_direct_loader:
            data = data.to(device)
            y_dir = data.y
            break
            
        for data in test_reverse_loader:
            data = data.to(device)
            y_rev = data.y
            break
            
        for data in test_loader:
            data = data.to(device)
            y_all = data.y
            break

    # Make predictions using each fold's model
    for i in range(fold):
        # Load model weights
        model.load_state_dict(torch.load(f"{weight_dir}/model_{i + 1}.pkl", map_location='cpu'))
        
        # Get predictions
        pred_all, y_all = evaluate(args, model, test_loader, device, return_tensor=True)
        pred_dir, y_dir = evaluate(args, model, test_direct_loader, device, return_tensor=True)
        pred_rev, y_rev = evaluate(args, model, test_reverse_loader, device, return_tensor=True)

        # Calculate metrics using the same format as main file
        corr_all, rmse_all, corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta = metrics(
            pred_all, pred_dir, pred_rev, y_all, y_dir, y_rev)

        logging.info(
            f'Fold {i + 1}, PCC: {corr_all:.3f}, RMSE: {rmse_all:.3f}, Direct PCC: {corr_dir:.3f}, Reverse PCC: {corr_rev:.3f}, Direct RMSE: {rmse_dir:.3f}, Reverse RMSE: {rmse_rev:.3f}')

        # Store predictions
        total_pred_dir.append(pred_dir.tolist())
        total_pred_rev.append(pred_rev.tolist())

        total_pred_all.append(pred_all.tolist())

    # Calculate average predictions across folds
    avg_pred_dir = torch.Tensor(total_pred_dir).mean(dim=0).to(device)
    # for data in test_direct_loader:
    #     print(avg_pred_dir,data.pos,data.mt)
    #     break
    avg_pred_rev = torch.Tensor(total_pred_rev).mean(dim=0).to(device)
    avg_pred_all = torch.Tensor(total_pred_all).mean(dim=0).to(device)
    
    Rfr, delta = compute_antisymmetry(avg_pred_dir, avg_pred_rev, y_dir)
    avg_corr_all, avg_rmse_all, avg_corr_dir, avg_rmse_dir, avg_corr_rev, avg_rmse_rev, avg_corr_dir_rev, avg_delta = metrics(
        avg_pred_all, avg_pred_dir, avg_pred_rev, y_all, y_dir, y_rev)


    logging.info(f'{avg_corr_all:.3f} {avg_rmse_all:.3f} {avg_corr_dir:.3f} {avg_corr_rev:.3f} {avg_rmse_dir:.3f} {avg_rmse_rev:.3f} Rfr:{Rfr:.3f} Delta:{delta:.3f}')



    if task=='test':
        task='Ssym'
    if task=='Ssym':
        plot_results_enhanced_v2(y_dir.cpu(), avg_pred_dir.cpu(), f'photo/sandian/{task}_dir.png', dataset=task+'_DIR', theme_color="#7313E9EB")
        plot_results_enhanced_v2(y_rev.cpu(), avg_pred_rev.cpu(), f'photo/sandian/{task}_rev.png', dataset=task+'_REV', theme_color='#7313E9EB')
        plot_results_enhanced_v2(y_all.cpu(), avg_pred_all.cpu(), f'photo/sandian/{task}.png', dataset=task, theme_color='#7313E9EB')
    if task=='S250':
        plot_results_enhanced_v2(y_dir.cpu(), avg_pred_dir.cpu(), f'photo/sandian/{task}_dir.png', dataset=task+'_DIR', theme_color="#EB2705")
        plot_results_enhanced_v2(y_rev.cpu(), avg_pred_rev.cpu(), f'photo/sandian/{task}_rev.png', dataset=task+'_REV', theme_color='#EB2705')
        plot_results_enhanced_v2(y_all.cpu(), avg_pred_all.cpu(), f'photo/sandian/{task}.png', dataset=task, theme_color='#EB2705')

    return avg_pred_all, avg_pred_dir, avg_pred_rev, y_all, y_dir, y_rev

def main():
    parser = argparse.ArgumentParser(description='ThermoGNN: predict and save results')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256,
                        help='input batch size for evaluation (default: 256)')
    parser.add_argument('--num-layer', type=int, dest='num_layer', default=5,
                        help='number of GNN message passing layers (default: 6)')
    parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=200,
                        help='embedding dimensions (default: 200)')
    parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="mean",
                        help='graph level pooling (sum, mean, max, attention)')
    parser.add_argument('--graph-dir', type=str, dest='graph_dir', 
                        default='data/Rgraph_80',
                        help='directory storing graphs data')
    parser.add_argument('--weight-dir', type=str, dest='weight_dir',
                        default='run/best',
                        help='directory containing model weights')
    parser.add_argument('--output-dir', type=str, dest='output_dir', default='predictions',
                        help='directory to save predictions')
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="rgcn",
                        help='gnn type (gin, gcn, gat, graphsage, rgcn)')
    parser.add_argument('--concat-type', type=str, dest='concat_type', default="concat",
                        help='concat type (lstm, bilstm, gru, concat)')
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")
    parser.add_argument('--num_relations', type=int, default=3,
                        help="Number of relation types (default: 3)")
    parser.add_argument('--feature-level', type=str, dest='feature_level', default='local',
                        help='global-local, global, or local')
    parser.add_argument('--contrast-curri', dest='contrast_curri', action='store_true', default=False,
                        help='using node contrast curriculum learning or not')
    parser.add_argument('--fds', type=bool, dest='fds', default=False,
                        help='Feature Distribution Smoothing')
    parser.add_argument('--loss', type=str, default='mse',
                        help='Loss function used during training')
    
    # 添加自蒸馏相关参数
    parser.add_argument('--self-distill', dest='self_distill', action='store_true', default=True,
                        help='Enable self distillation training')
    parser.add_argument('--loss-coefficient', type=float, dest='loss_coefficient', default=0.3,
                        help='Loss coefficient for KL divergence in self distillation (default: 0.3)')
    parser.add_argument('--feature-loss-coefficient', type=float, dest='feature_loss_coefficient', default=0.05,
                        help='Feature loss coefficient for L2 loss in self distillation (default: 0.03)')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for knowledge distillation (default: 3.0)')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", 
        datefmt="%F %A %T", 
        level=logging.INFO
    )

    # Initialize model with parameters matching main file
    input_dim = 200  # 修改input_dim从60到80
    model = GraphGNN(
        num_layer=args.num_layer, 
        input_dim=input_dim, 
        emb_dim=args.emb_dim, 
        out_dim=1, 
        JK="last",
        drop_ratio=args.dropout_ratio, 
        graph_pooling=args.graph_pooling, 
        gnn_type=args.gnn_type,
        concat_type=args.concat_type, 
        fds=args.fds, 
        feature_level=args.feature_level, 
        contrast_curri=args.contrast_curri,
        num_relations=args.num_relations,  # 添加num_relations参数
        self_distill=args.self_distill      # 添加self_distill参数
    )

    # Run prediction on different test setss
    # predict_and_save(args, model, "test", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "myoglobin", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "p53", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "S250", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "S350", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "S879", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "S605", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "S1925", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "S669", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "Tp53_test", args.graph_dir, args.weight_dir, fold=args.split)
    # predict_and_save(args, model, "1T15case", args.graph_dir, args.weight_dir, fold=args.split)


    # TASK_LIST = ["test", "myoglobin", "p53", "S250", "S350", "S879", "S605", "S1925"]
    TASK_LIST = ["S350","S605","S1925","myoglobin","S250","test",  "p53", "S879" ]
    # TASK_LIST = ["S1925","myoglobin"]
    # TASK_LIST = ["p53"]

    true_dict, pred_dict = {}, {}  # 收集所有任务结果

    for task in TASK_LIST:
        try:
            avg_pred_all, avg_pred_dir, avg_pred_rev, y_all, y_dir, y_rev = \
                predict_and_save(args, model, task, args.graph_dir, args.weight_dir,
                                 fold=args.split, color=None)  # color 不用了，后面统一配
            if task=="test":
                task='Ssym'
            true_dict[task] = y_all.cpu().numpy()
            pred_dict[task] = avg_pred_all.cpu().numpy()
            logging.info(f"Task {task} 预测完成，已收集数据。")
        except Exception as e:
            logging.warning(f"Task {task} 失败，已跳过：{e}")
            continue

if __name__ == "__main__":
    main()
