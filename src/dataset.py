from collections import Counter

import torch
import pickle as pkl
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from src.utils.weights import assign_weights

import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window


def get_bin_idx(x):
    return max(min(int(x * np.float32(5)), 12), -12)


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key == 'wide_nodes':
            return self.x_s.num_nodes
        if key == 'mut_nodes':
            return self.x_t.num_nodes
        else:
            return super().__inc__(key, value, *args)


def load_dataset(graph_dir, split, labeled=True, dir=False):
    data_list = []
    num_nodes = 0
    num_edges = 0

    cos_file = open('cos.txt', 'w')

    for i, name in enumerate(open(f"data/R3_names/{split}_names.txt")):
        name = name.strip()

        # 加载字典格式的数据
        with open(f"{graph_dir}/{split}/{name}_wt_rgcn.pkl", 'rb') as f:
            G_wt_data = pkl.load(f)
        with open(f"{graph_dir}/{split}/{name}_mut_rgcn.pkl", 'rb') as f:
            G_mut_data = pkl.load(f)

        # 直接从字典创建Data对象
        data_wt = Data(
            x=G_wt_data['x'],
            edge_index=G_wt_data['edge_index'],
            edge_type=G_wt_data['edge_type']
        )
        data_mut = Data(
            x=G_mut_data['x'],
            edge_index=G_mut_data['edge_index'],
            edge_type=G_mut_data['edge_type']
        )

        wt_node_count = G_wt_data['num_nodes']
        mut_node_count = G_mut_data['num_nodes']

        mask = data_wt.edge_type !=4
        # mask = data_wt.edge_type ==1
        edge_index_wt = data_wt.edge_index[:, mask]  # 直接筛选列
        edge_type_wt = data_wt.edge_type[mask]  # 筛选对应的边类型

        mask = data_mut.edge_type !=4
        # mask = data_mut.edge_type ==1
        edge_index_mut = data_mut.edge_index[:, mask]  # 直接筛选列
        edge_type_mut = data_mut.edge_type[mask]  # 筛选对应的边类型




        # data_direct = PairData(data_wt.edge_index, data_wt.x,
        #                        data_mut.edge_index, data_mut.x)

        data_direct = PairData(edge_index_wt, data_wt.x,
                               edge_index_mut, data_mut.x)

        data_direct.wide_res_idx = G_wt_data['mut_pos']
        data_direct.mut_res_idx = G_mut_data['mut_pos']
        # data_direct.edge_r_s = data_wt.edge_type
        # data_direct.edge_r_t = data_mut.edge_type

        data_direct.edge_r_s = edge_type_wt
        data_direct.edge_r_t = edge_type_mut

        data_direct.wt_count = wt_node_count
        data_direct.mut_count = mut_node_count

        # data_reverse = PairData(data_mut.edge_index, data_mut.x,
        #                         data_wt.edge_index, data_wt.x)

        data_reverse = PairData(edge_index_mut,data_mut.x,
                                edge_index_wt, data_wt.x)

        data_reverse.wide_res_idx = G_mut_data['mut_pos']
        data_reverse.mut_res_idx = G_wt_data['mut_pos']

        # data_reverse.edge_r_s = data_mut.edge_type
        # data_reverse.edge_r_t = data_wt.edge_type

        data_reverse.edge_r_s =edge_type_mut
        data_reverse.edge_r_t =edge_type_wt

        data_reverse.wt_count = mut_node_count
        data_reverse.mut_count = wt_node_count

        if labeled:
            if split == 'S669':
                data_direct.y = -G_wt_data['y']
                data_reverse.y = G_mut_data['y']
            else:
                data_direct.y = G_wt_data['y']
                data_reverse.y = -G_mut_data['y']

        if dir:
            weights = assign_weights("data/datasets/train_data_noisy.txt")
            data_direct.wy = torch.tensor(weights[i])
            data_reverse.wy = torch.tensor(weights[i])

        data_list.append(data_direct)
        data_list.append(data_reverse)
        num_nodes += wt_node_count + mut_node_count
        num_edges += G_wt_data['relation_info']['total_edges'] + G_mut_data['relation_info']['total_edges']

    print(f'{split.upper()} DATASET:')
    print(f'Number of nodes: {num_nodes / len(data_list):.2f}')
    print(f'Number of edges: {num_edges / len(data_list):.2f}')
    print(f'Average node degree: {num_edges / num_nodes:.2f}')

    return data_list
