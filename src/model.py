import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import batch_norm
from torch_geometric.nn import GraphConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, GraphNorm, global_add_pool, global_max_pool, GlobalAttention,GatedGraphConv
from src.utils.fds import FDS
from rgcn_gate import RGCNConv


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


class GNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim=1, JK="last", drop_ratio=0, gnn_type="gin", num_relations=8, self_distill=True):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.heads = 4
        self.gnn_type = gnn_type
        self.self_distill = self_distill
        self.out_dim = out_dim
        
        # GNN layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            in_dim = input_dim if layer == 0 else emb_dim
            if gnn_type == "gin":
                self.gnns.append(GINConv(nn.Sequential(nn.Linear(in_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
                                                       nn.Linear(emb_dim, emb_dim))))
            elif gnn_type == "gcn":
                self.gnns.append(GraphConv(in_dim, emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(in_dim, emb_dim))
            elif gnn_type == "rgcn":
                self.gnns.append(RGCNConv(
                    in_channels=in_dim,
                    out_channels=emb_dim,
                    num_relations=num_relations,
                    aggr='mean',
                    root_weight=True,
                    bias=True,
                    gate_type='vector'


                    # num_heads=2,
                    # heads=8,
                    # attention_mode='learnable_balance'
                ))
                """
                RGCN with enhanced intra-relation attention mechanism.

                This implementation provides three aggregation modes:
                1. 'neighbor_weighted': alpha + 1/neighbor_count
                2. 'renormalized': (alpha + 1) with renormalization  
                3. 'learnable_balance': lambda * alpha + 1/neighbor_count
                """
            elif gnn_type == "nsagat":
                self.gnns.append(NSAGATConv(
                    in_dim, emb_dim,
                    heads=self.heads,
                    concat=False,
                    block_size=8, stride=4,
                    select_block_size=8, num_select_blocks=2,
                    window_size=10
                ))
            elif gnn_type == "graphsage":
                self.gnns.append(SAGEConv(in_dim, emb_dim))
            else:
                print(f'no_{gnn_type}')
                raise ValueError(f"Invalid GNN type")
        
        # 保留原有的跳跃连接相关层
        # self.fc1 = nn.Linear(input_dim, emb_dim)
        self.fc1 = nn.Linear(80, emb_dim)
        self.fc2 = nn.Linear(200, 200)

    def forward(self, x, edge_index, mut_res_idx, batch, edge_attr=None, edge_type=None):
        h_list = [self.fc1(x)]
        # h_list = [x]
        # 存储每层的特征和预测结果（用于自蒸馏）
        layer_features = []  # 用于特征蒸馏
        layer_predictions = []  # 用于KL蒸馏
        pooled_features = []  # 用于最终预测
        features = []
        
        for layer in range(self.num_layer):
            # 前向传播
            if self.gnn_type == "rgcn":
                if edge_type is None:
                    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                h = self.gnns[layer](h_list[layer], edge_index, edge_type)
            else:
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            
            h_list.append(h)

            if len(h_list) == 2:
                h_list[-1] = h_list[-1] + self.fc1(x)
            if len(h_list) >= 3:
                h_list[-1] = h_list[-1] + h_list[-2]
                h_list[-1] = h_list[-1] + self.fc1(x)
            if self.self_distill:
                # 获取当前层特征
                current_h = h_list[-1]

                # 通过bottleneck层
                # bottleneck_feature = self.bottlenecks[layer](current_h)
                layer_features.append(current_h)
                # # 池化操作（简化版本，使用平均池化）
                # # 这里需要根据您的具体需求调整池化方式
                mut_res_idx = mut_res_idx.to(current_h.device)
                pooled_h = current_h[mut_res_idx]
                pooled_features.append(pooled_h)
        # 返回结果
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)

        if self.self_distill:
            return layer_features, pooled_features
        else:
            return node_representation


# 保持原有的初始化函数
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)


def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4: l // 2].data.fill_(1.0)
        hh_b[l // 4: l // 2].data.fill_(1.0)


class GraphGNN(nn.Module):
    def __init__(self, num_layer, input_dim, emb_dim, out_dim, JK="last", drop_ratio=0, graph_pooling="attention",
                 gnn_type="gat", concat_type='lstm', fds=False, feature_level='both', contrast_curri=False, 
                 num_relations=4, self_distill=True) -> object:
        super(GraphGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.concat_type = concat_type
        self.feature_level = feature_level
        self.contrast_curri = contrast_curri
        self.self_distill = self_distill
        
        # 原有的注意力层
        self.global_local_att0 = nn.Linear(400, 200)
        self.global_local_att1 = nn.Linear(400, 200)

        # 处理LSTM相关配置
        if self.concat_type == 'lstm':
            self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1)
            if not self.self_distill:  # 只在非自蒸馏模式下使用原始FC
                self.fc = nn.Linear(self.emb_dim, self.out_dim)
        elif self.concat_type == 'bilstm':
            self.lstm_graph = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, num_layers=1,
                                      bidirectional=True)
            init_lstm_orth(self.lstm_graph)
            if not self.self_distill:
                self.fc = nn.Linear(2 * self.emb_dim, self.out_dim)
        else:
            if not self.self_distill:  # 只在非自蒸馏模式下创建FC层
                if self.feature_level == 'global-local':
                    self.fc = nn.Sequential(
                        nn.Linear(2* self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                        nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                        nn.Linear(self.emb_dim, self.out_dim))
                else:
                    self.fc = nn.Sequential(
                        nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                        nn.Linear(self.emb_dim, 100), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                        nn.Linear(100, self.out_dim))

        if fds:
            self.dir = True
            self.FDS = FDS(4 * self.emb_dim)
        else:
            self.dir = False

        # 创建GNN，传递self_distill参数
        self.gnn = GNN(num_layer, input_dim, emb_dim, out_dim, JK, drop_ratio, 
                      gnn_type=gnn_type, self_distill=self_distill)

        # 池化层
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
        
        # 自蒸馏模式下的最终分类器
        if self.self_distill:
            if self.feature_level == 'global-local':
                self.final_fc = nn.Sequential(
                    nn.Linear(2* self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                    nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                    nn.Linear(self.emb_dim, self.out_dim))
            else:
                self.final_fc = nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                    nn.Linear(self.emb_dim, 100), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                    nn.Linear(100, self.out_dim))
            # 每层的分类器
            self.classifiers = nn.ModuleList()
            for layer in range(num_layer):
                # 分类器：用于预测
                classifier=nn.Sequential(
                        nn.Linear(self.emb_dim, self.emb_dim), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                        nn.Linear(self.emb_dim, 100), nn.LeakyReLU(0.1), nn.Dropout(p=self.drop_ratio),
                        nn.Linear(100, self.out_dim))
                self.classifiers.append(classifier)

    def forward_once(self, x, edge_index, batch, mut_res_idx, edge_attr=None, edge_type=None):
        mut_res_idx = torch.tensor([mut_res_idx]).cuda()
        
        if self.self_distill:
            layer_features, pooled_features = self.gnn(
                x, edge_index, mut_res_idx, batch, edge_attr=edge_attr, edge_type=edge_type)

            # 使用最后一层的池化特征作为图表示
            graph_rep = pooled_features[-1] if pooled_features else self.pool(layer_features[-1], batch)
            
            # 获取突变位点表示
            mut_node_rep=layer_features[-1]
            
            return graph_rep, mut_node_rep,layer_features, pooled_features
        else:
            node_representation = self.gnn(x, edge_index, mut_res_idx, batch, edge_attr=edge_attr, edge_type=edge_type)
            graph_rep = self.pool(node_representation, batch)
            mut_node_rep = node_representation[mut_res_idx].squeeze(0)
            
            return graph_rep, mut_node_rep

    def forward(self, data, epoch=0):
        # 处理索引
        wide_res_idx = []
        mut_res_idx = []
        wt_idx = 0
        for i in range(len(data.wide_res_idx)):
            wide_res_idx.append(data.wide_res_idx[i].item() + wt_idx)
            wt_idx += data.wt_count[i].item()

        mut_idx = 0
        for i in range(len(data.mut_res_idx)):
            mut_res_idx.append(data.mut_res_idx[i].item() + mut_idx)
            mut_idx += data.mut_count[i].item()

        # 获取edge_type
        edge_type_s = data.edge_r_s
        edge_type_t = data.edge_r_t

        if self.self_distill:
            # 自蒸馏模式
            graph_rep_be, node_rep_be,  layer_feat_be, node_feat_be = self.forward_once(
                data.x_s, data.edge_index_s, data.x_s_batch, wide_res_idx,
                edge_attr=getattr(data, 'edge_attr_s', None), edge_type=edge_type_s
            )
            graph_rep_af, node_rep_af,  layer_feat_af, node_feat_af = self.forward_once(
                data.x_t, data.edge_index_t, data.x_t_batch, mut_res_idx,
                edge_attr=getattr(data, 'edge_attr_t', None), edge_type=edge_type_t
            )
            all_layer_preds = []
            # 计算差异特征
            if self.feature_level == 'global-local':
                x1 = node_rep_be - node_rep_af
                x2 = graph_rep_be - graph_rep_af
                x = torch.cat([x1, x2], dim=1)
            elif self.feature_level == 'local':
                cluster = []
                num=len(layer_feat_af)
                for i in range(num):
                    x=node_feat_be[i] - node_feat_af[i]
                    cluster.append(x.squeeze())
                    pre=self.classifiers[i](x)
                    pre=pre.squeeze()
                    all_layer_preds.append(pre)

                # for i in range(num):
                #     x=layer_feat_be[i]-layer_feat_af[i]
                #     cluster.append(x)

            
            # return all_layer_preds, layer_feat_be, layer_feat_af
            return all_layer_preds, cluster, layer_feat_af
            
        else:
            # 原始模式
            graph_rep_be, node_rep_be = self.forward_once(
                data.x_s, data.edge_index_s, data.x_s_batch, wide_res_idx,
                edge_attr=getattr(data, 'edge_attr_s', None), edge_type=edge_type_s
            )
            graph_rep_af, node_rep_af = self.forward_once(
                data.x_t, data.edge_index_t, data.x_t_batch, mut_res_idx,
                edge_attr=getattr(data, 'edge_attr_t', None), edge_type=edge_type_t
            )

            # 原有的特征融合逻辑...
            if self.concat_type == 'concat':
                if self.feature_level == 'global-local':
                    x1 = node_rep_be - node_rep_af
                    x2 = graph_rep_be - graph_rep_af
                    x = torch.cat([x1, x2], dim=1)
                elif self.feature_level == 'local':
                    x = node_rep_be - node_rep_af
            else:
                # LSTM处理逻辑...
                graph_rep_0, graph_rep_1 = graph_rep_be.unsqueeze_(0), graph_rep_af.unsqueeze_(0)
                lstm_graph_in = torch.cat((graph_rep_0, graph_rep_1), dim=0)
                graph_t1, (_, _) = self.lstm_graph(lstm_graph_in)
                x = graph_t1[-1]

            if self.dir:
                smooth_x = x
                x = self.FDS.smooth(smooth_x, data.y, epoch)
                x = self.fc(x)
                return torch.squeeze(x), smooth_x
            elif self.contrast_curri:
                similarity_list = []
                for i in range(node_rep_be.shape[0]):
                    similarity_list.append(cosine_similarity(np.asarray(node_rep_be[i].cpu().detach()),
                                                             np.asarray(node_rep_af[i].cpu().detach())))
                x = self.fc(x)
                return torch.squeeze(x), similarity_list
            else:
                x = self.fc(x)
                return torch.squeeze(x)
