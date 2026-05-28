import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import torch
from Bio.PDB import PDBParser, is_aa
from sklearn.neighbors import NearestNeighbors

from src.utils.features import get_node_feature, read_hhm_file, read_scoring_functions,read_pssm_file,get_node_feature_with_pssm


def get_CA(res):
    return res["CA"]


class RGCNMultiRelationGraphBuilder:
    """RGCN多关系图构建器 - PyTorch Geometric格式"""

    def __init__(self):
        self.relation_types = {
            'sequence': 0,  # 序列边 (双向)
            'spatial_0_4': 1,  # 空间边 0-4埃米 (双向)
            'spatial_4_6': 2,  # 空间边 4-6埃米 (双向)
            'spatial_6_8': 3,  # 空间边 6-8埃米 (双向)
            'knn_1_5': 4,  # KNN边 1-5最近 (单向)
            'knn_6_10': 5,  # KNN边 6-10最近 (单向)
            'knn_11_15': 6,  # KNN边 11-15最近 (单向)
            'self_loop': 7  # 自环 (单向)
        }
        self.num_relations = len(self.relation_types)

    def build_sequence_relations(self, nodes_list, seq_window=2):
        """
        构建序列关系矩阵 (双向)

        Args:
            nodes_list: 节点列表（残基ID）
            seq_window: 序列窗口大小

        Returns:
            sequence_edges: 序列边列表 [(i, j), ...]
        """
        sequence_edges = []
        sorted_nodes = sorted(nodes_list)
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

        for i, res_i in enumerate(sorted_nodes):
            for j in range(i + 1, len(sorted_nodes)):
                res_j = sorted_nodes[j]
                seq_distance = abs(res_j - res_i)

                if seq_distance <= seq_window:
                    idx_i = node_to_idx[res_i]
                    idx_j = node_to_idx[res_j]
                    # 双向边
                    sequence_edges.extend([(idx_i, idx_j), (idx_j, idx_i)])
                else:
                    break

        return sequence_edges

    def build_spatial_relations(self, chain, nodes_list, contact_threshold=8):
        """
        构建空间关系矩阵 - 按距离分为三种关系类型 (双向)

        Args:
            chain: BioPython链对象
            nodes_list: 节点列表
            contact_threshold: 最大距离阈值（默认8埃米）

        Returns:
            spatial_edges_dict: 包含三种空间关系的字典
                - 'spatial_0_4': 0-4埃米的边
                - 'spatial_4_6': 4-6埃米的边  
                - 'spatial_6_8': 6-8埃米的边
        """
        spatial_edges_dict = {
            'spatial_0_4': [],
            'spatial_4_6': [],
            'spatial_6_8': []
        }
        
        num_nodes = len(nodes_list)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                res_i = nodes_list[i]
                res_j = nodes_list[j]
                distance = get_CA(chain[res_i]) - get_CA(chain[res_j])

                if distance <= contact_threshold:
                    # 按距离分类，重叠时归入最小范围
                    if distance <= 4.0:
                        spatial_edges_dict['spatial_0_4'].extend([(i, j), (j, i)])
                    elif distance <= 6.0:
                        spatial_edges_dict['spatial_4_6'].extend([(i, j), (j, i)])
                    elif distance <= 8.0:
                        spatial_edges_dict['spatial_6_8'].extend([(i, j), (j, i)])

        return spatial_edges_dict

    def build_knn_relations(self, chain, nodes_list, k=15):
        """
        构建KNN关系矩阵 - 按邻居排序分为三种关系类型 (单向)

        Args:
            chain: BioPython链对象
            nodes_list: 节点列表
            k: KNN的K值（默认15）

        Returns:
            knn_edges_dict: 包含三种KNN关系的字典
                - 'knn_1_5': 1-5最近邻的边
                - 'knn_6_10': 6-10最近邻的边
                - 'knn_11_15': 11-15最近邻的边
        """
        knn_edges_dict = {
            'knn_1_5': [],
            'knn_6_10': [],
            'knn_11_15': []
        }
        
        num_nodes = len(nodes_list)

        # 提取CA原子坐标
        coordinates = []
        for res_id in nodes_list:
            ca_coord = get_CA(chain[res_id]).get_coord()
            coordinates.append(ca_coord)

        coordinates = np.array(coordinates)

        # KNN计算
        k_actual = min(k + 1, num_nodes)  # +1因为包括自己
        nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='auto').fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

        # 构建KNN边 - 按邻居排序分类
        for i in range(num_nodes):
            neighbors = indices[i][1:]  # 跳过自己
            
            # 1-5最近邻
            for j in range(min(5, len(neighbors))):
                neighbor_idx = neighbors[j]
                knn_edges_dict['knn_1_5'].append((i, neighbor_idx))
            
            # 6-10最近邻
            for j in range(5, min(10, len(neighbors))):
                neighbor_idx = neighbors[j]
                knn_edges_dict['knn_6_10'].append((i, neighbor_idx))
            
            # 11-15最近邻
            for j in range(10, min(15, len(neighbors))):
                neighbor_idx = neighbors[j]
                knn_edges_dict['knn_11_15'].append((i, neighbor_idx))

        return knn_edges_dict

    def build_self_loop_relations(self, num_nodes):
        """
        构建自环关系 (单向)

        Args:
            num_nodes: 节点数量

        Returns:
            self_loop_edges: 自环边列表 [(i, i), ...]
        """
        self_loop_edges = [(i, i) for i in range(num_nodes)]
        return self_loop_edges

    def create_edge_data(self, num_nodes, sequence_edges, spatial_edges_dict, knn_edges_dict):
        """
        创建PyTorch Geometric格式的边数据

        Args:
            num_nodes: 节点数量
            sequence_edges: 序列边列表
            spatial_edges_dict: 空间边字典
            knn_edges_dict: KNN边字典

        Returns:
            edge_index: (2, num_edges) 张量
            edge_type: (num_edges,) 张量
        """
        all_edges = []
        all_edge_types = []

        # 1. 序列关系边
        for edge in sequence_edges:
            all_edges.append(edge)
            all_edge_types.append(self.relation_types['sequence'])

        # 2. 空间关系边（三种类型）
        for spatial_type, edges in spatial_edges_dict.items():
            for edge in edges:
                all_edges.append(edge)
                all_edge_types.append(self.relation_types[spatial_type])

        # 3. KNN关系边（三种类型）
        for knn_type, edges in knn_edges_dict.items():
            for edge in edges:
                all_edges.append(edge)
                all_edge_types.append(self.relation_types[knn_type])

        # 4. 自环边
        self_loop_edges = self.build_self_loop_relations(num_nodes)
        for edge in self_loop_edges:
            all_edges.append(edge)
            all_edge_types.append(self.relation_types['self_loop'])

        # 转换为张量格式
        if len(all_edges) > 0:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()  # (2, num_edges)
            edge_type = torch.tensor(all_edge_types, dtype=torch.long)  # (num_edges,)
        else:
            # 如果没有边，创建空张量
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty(0, dtype=torch.long)

        return edge_index, edge_type

    def print_relation_statistics(self, sequence_edges, spatial_edges_dict, knn_edges_dict, num_nodes):
        """打印关系统计信息"""
        self_loop_edges = self.build_self_loop_relations(num_nodes)

        print(f"图统计信息:")
        print(f"  节点数量: {num_nodes}")
        print(f"  序列边数量: {len(sequence_edges)} (双向)")
        
        # 打印空间边统计
        total_spatial = 0
        for spatial_type, edges in spatial_edges_dict.items():
            print(f"  {spatial_type}边数量: {len(edges)} (双向)")
            total_spatial += len(edges)
        
        # 打印KNN边统计
        total_knn = 0
        for knn_type, edges in knn_edges_dict.items():
            print(f"  {knn_type}边数量: {len(edges)} (单向)")
            total_knn += len(edges)
            
        print(f"  自环边数量: {len(self_loop_edges)} (单向)")
        
        total_edges = len(sequence_edges) + total_spatial + total_knn + len(self_loop_edges)
        print(f"  总边数量: {total_edges}")
        print(f"  关系类型数量: {self.num_relations}")

        # 检查边的重叠情况（仅在不同大类之间）
        sequence_set = set(sequence_edges)
        all_spatial_set = set()
        for edges in spatial_edges_dict.values():
            all_spatial_set.update(edges)
        
        all_knn_set = set()
        for edges in knn_edges_dict.values():
            all_knn_set.update(edges)

        seq_spatial_overlap = len(sequence_set & all_spatial_set)
        seq_knn_overlap = len(sequence_set & all_knn_set)
        spatial_knn_overlap = len(all_spatial_set & all_knn_set)
        all_overlap = len(sequence_set & all_spatial_set & all_knn_set)

        print(f"  关系重叠:")
        print(f"    序列-空间重叠: {seq_spatial_overlap}")
        print(f"    序列-KNN重叠: {seq_knn_overlap}")
        print(f"    空间-KNN重叠: {spatial_knn_overlap}")
        print(f"    三种关系重叠: {all_overlap}")


def make_rgcn_graph(record, aa_features, out_dir, is_wt=True, split="train",
                    contact_threshold=8, local_radius=12, k=15, seq_window=2):
    """
    构建RGCN多关系图 - PyTorch Geometric格式

    Args:
        record: 数据记录
        aa_features: 氨基酸特征
        out_dir: 输出目录
        is_wt: 是否为野生型
        split: 数据集分割
        contact_threshold: 空间距离阈值（默认改为8埃米）
        local_radius: 局部区域半径
        k: KNN的K值（默认改为15）
        seq_window: 序列窗口大小
    """

    # 解析记录
    if len(record.strip().split()) == 5:  # with known ddG
        pdb_name, mut_pos, wt, mut, ddG = record.strip().split()
        ddG = float(ddG)
    else:
        pdb_name, mut_pos, wt, mut = record.strip().split()
        ddG = None

    p = PDBParser()
    pdb_id, chain_id = pdb_name[:-1], pdb_name[-1]
    mut_pos = int(mut_pos)

    # 构建文件路径
    if is_wt:
        suffix = pdb_name
        out_path = f"{out_dir}/{split}/{pdb_name}_{wt}{mut_pos}{mut}_wt_rgcn.pkl"
    else:
        suffix = f"{pdb_name}_{wt}{mut_pos}{mut}"
        out_path = f"{out_dir}/{split}/{pdb_name}_{wt}{mut_pos}{mut}_mut_rgcn.pkl"

    pdb_path = f"data/pdbs/{split}/{pdb_name}/{suffix}_relaxed.pdb"
    hhm_path = f"data/hhm/{split}/{suffix}.hhm"
    pssm_path = f"data/pssm/{split}/{suffix}.pssm"
    pssm_faeature=read_pssm_file(pssm_path)

    print(f"处理: {pdb_name}, 突变: {wt}{mut_pos}{mut}, 类型: {'WT' if is_wt else 'MUT'}")

    # 解析蛋白质结构
    structure = p.get_structure(pdb_name, pdb_path)
    chain = structure[0][chain_id]

    mut_res = chain[mut_pos]
    mut_center = get_CA(mut_res)

    # 选择局部区域内的残基
    nodes_list = []
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            center = get_CA(res)
            distance = center - mut_center
            if distance <= local_radius:
                nodes_list.append(res.id[1])

    num_nodes = len(nodes_list)
    mut_index = nodes_list.index(mut_res.id[1])

    print(f"局部区域节点数: {num_nodes}")

    # 创建图构建器
    builder = RGCNMultiRelationGraphBuilder()

    # 构建各种关系的边
    print("构建多关系边...")
    sequence_edges = builder.build_sequence_relations(nodes_list, seq_window)
    spatial_edges_dict = builder.build_spatial_relations(chain, nodes_list, contact_threshold)
    knn_edges_dict = builder.build_knn_relations(chain, nodes_list, k)

    # 打印统计信息
    builder.print_relation_statistics(sequence_edges, spatial_edges_dict, knn_edges_dict, num_nodes)

    # 创建PyTorch Geometric格式的边数据
    edge_index, edge_type = builder.create_edge_data(
        num_nodes, sequence_edges, spatial_edges_dict, knn_edges_dict)

    print(f"  边索引形状: {edge_index.shape}")
    print(f"  边类型形状: {edge_type.shape}")

    # 读取节点特征
    mat = read_hhm_file(hhm_path)
    scoring = read_scoring_functions(pdb_path)
    # features = get_node_feature(nodes_list, mat, scoring, aa_features, chain)
    features=get_node_feature_with_pssm(nodes_list, mat, scoring, aa_features,pssm_faeature,chain)

    # 创建节点特征张量
    X = torch.tensor(features, dtype=torch.float32)

    # 创建标签 (如果有ddG值)
    if ddG is not None:
        # 回归任务：直接使用ddG值
        y = torch.tensor([ddG], dtype=torch.float32)
    else:
        y = None

    # 保存RGCN格式的数据
    rgcn_data = {
        'edge_index': edge_index,  # (2, num_edges)
        'edge_type': edge_type,  # (num_edges,)
        'x': X,  # (num_nodes, num_features)
        'y': y,  # (1,) for regression
        'mut_pos': mut_index,  # 突变位置索引
        'nodes_list': nodes_list,  # 原始节点列表
        'ddG': ddG,  # 原始ddG值
        'num_nodes': num_nodes,  # 节点数量
        'num_relations': builder.num_relations,  # 关系类型数量
        'relation_types': builder.relation_types,  # 关系类型映射
        'relation_info': {
            'sequence': len(sequence_edges),
            'spatial_0_4': len(spatial_edges_dict['spatial_0_4']),
            'spatial_4_6': len(spatial_edges_dict['spatial_4_6']),
            'spatial_6_8': len(spatial_edges_dict['spatial_6_8']),
            'knn_1_5': len(knn_edges_dict['knn_1_5']),
            'knn_6_10': len(knn_edges_dict['knn_6_10']),
            'knn_11_15': len(knn_edges_dict['knn_11_15']),
            'self_loop': num_nodes,
            'total_edges': edge_index.shape[1]
        }
    }

    # 保存数据
    with open(out_path, 'wb') as f:
        pkl.dump(rgcn_data, f, pkl.HIGHEST_PROTOCOL)

    print(f"RGCN图数据已保存到: {out_path}")

    return rgcn_data


def batch_make_rgcn_graphs(records, aa_features, out_dir, split="train", **kwargs):
    """
    批量构建RGCN图

    Args:
        records: 数据记录列表
        aa_features: 氨基酸特征
        out_dir: 输出目录
        split: 数据集分割
        **kwargs: 其他参数
    """
    print(f"批量构建RGCN图，共 {len(records)} 条记录")

    all_data = []
    for i, record in enumerate(records):
        print(f"\n处理记录 {i + 1}/{len(records)}: {record.strip()}")

        # 野生型
        wt_data = make_rgcn_graph(record, aa_features, out_dir,
                                  is_wt=True, split=split, **kwargs)
        all_data.append(wt_data)

        # 突变型
        mut_data = make_rgcn_graph(record, aa_features, out_dir,
                                   is_wt=False, split=split, **kwargs)
        all_data.append(mut_data)

    print(f"\n批量处理完成，生成 {len(all_data)} 个图")
    return all_data
