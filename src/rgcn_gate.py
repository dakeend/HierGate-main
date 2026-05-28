from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

import torch_geometric.backend
import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm
from torch.nn import functional as F


def masked_edge_index(edge_index: Adj, edge_mask: Tensor) -> Adj:
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGCNConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int,
            num_bases: Optional[int] = None,
            num_blocks: Optional[int] = None,
            aggr: str = 'mean',
            root_weight: bool = True,
            is_sorted: bool = False,
            bias: bool = True,
            use_relation_gate: bool = True,  # 新增：是否使用关系门控
            gate_type: str = 'scalar',  # 'scalar'或'vector'
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted
        self.use_relation_gate = use_relation_gate
        self.gate_type = gate_type

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self._use_segment_matmul_heuristic_output: Optional[bool] = None

        # 原有的权重参数
        if num_bases is not None:
            self.weight = Parameter(
                torch.empty(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.empty(num_relations, num_blocks,
                            in_channels[0] // num_blocks,
                            out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.empty(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        # Alibaba风格的关系门控参数
        if self.use_relation_gate:
            if self.gate_type == 'scalar':
                # 标量门控：每个关系一个标量值
                self.relation_gates = Parameter(torch.empty(num_relations, 1))
            elif self.gate_type == 'node':
                self.relation_gates = Parameter(torch.empty(out_channels, 1))
            else:  # vector
                # 向量门控：每个关系一个向量（逐元素门控）
                self.relation_gates = Parameter(torch.empty(num_relations, out_channels))

            # 可选的全局门控偏置
            self.gate_bias = Parameter(torch.zeros(1))

        if root_weight:
            self.root = Parameter(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # 原有的关系间注意力参数
        self.attention = Parameter(torch.empty(out_channels, 1))
        # self.attention1 = Parameter(torch.empty(out_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)
        glorot(self.attention)

        # 初始化关系门控参数
        if self.use_relation_gate:
            if self.gate_type == 'scalar':
                # 初始化为接近1的值，保持大部分信息流动
                torch.nn.init.constant_(self.relation_gates, 1.0)
            else:
                glorot(self.relation_gates)
            zeros(self.gate_bias)

    def apply_relation_gate(self, features: Tensor, relation_idx: int) -> Tensor:
        """应用关系特定门控"""
        if not self.use_relation_gate:
            return features

        if self.gate_type == 'scalar':
            # 标量门控：整个特征乘以一个标量
            gate = torch.sigmoid(self.relation_gates[relation_idx] + self.gate_bias)
            return features * gate
        elif self.gate_type == 'node':
            gate = torch.sigmoid(features*self.relation_gates[relation_idx])
            return features * gate
        else:
            # 向量门控：每个维度独立门控
            gate = torch.sigmoid(self.relation_gates[relation_idx] + self.gate_bias)
            # print(gate)
            return features * gate.unsqueeze(0)  # 广播到所有节点

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # 初始化输出
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====
            if not torch.is_floating_point(x_r) and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            # 收集所有关系类型的特征
            relation_features = []
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                h = h.contiguous().view(-1, self.out_channels)

                # 应用关系门控
                h = self.apply_relation_gate(h, i)

                relation_features.append(h)

            # 计算注意力权重（关系间门控）
            if len(relation_features) > 0:
                stacked_features = torch.stack(relation_features, dim=0)
                attention_scores = torch.matmul(stacked_features, self.attention)
                attention_weights = F.softmax(attention_scores, dim=0)
                out = torch.sum(attention_weights * stacked_features, dim=0)
                # out = 2*torch.mean(stacked_features, dim=0)

        else:  # No regularization/Basis-decomposition ========================
            use_segment_matmul = torch_geometric.backend.use_segment_matmul
            if use_segment_matmul is None:
                segment_count = scatter(torch.ones_like(edge_type), edge_type,
                                        dim_size=self.num_relations)
                self._use_segment_matmul_heuristic_output = (
                    torch_geometric.backend.use_segment_matmul_heuristic(
                        num_segments=self.num_relations,
                        max_segment_size=int(segment_count.max()),
                        in_channels=self.weight.size(1),
                        out_channels=self.weight.size(2),
                    ))
                assert self._use_segment_matmul_heuristic_output is not None
                use_segment_matmul = self._use_segment_matmul_heuristic_output

            # 如果使用关系门控，禁用segment_matmul优化以确保正确应用门控
            if (use_segment_matmul and torch_geometric.typing.WITH_SEGMM
                    and not is_compiling() and self.num_bases is None
                    and x_l.is_floating_point()
                    and isinstance(edge_index, Tensor)
                    and not self.use_relation_gate):  # 关键修改：使用门控时禁用优化

                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = index_sort(
                            edge_type, max_value=self.num_relations)
                        edge_index = edge_index[:, perm]
                edge_type_ptr = index2ptr(edge_type, self.num_relations)
                out = self.propagate(edge_index, x=x_l,
                                     edge_type_ptr=edge_type_ptr, size=size)
            else:
                # 关系循环方式：收集所有关系类型的特征
                relation_features = []
                for i in range(self.num_relations):
                    tmp = masked_edge_index(edge_index, edge_type == i)

                    if not torch.is_floating_point(x_r):
                        h = self.propagate(tmp, x=weight[i, x_l], edge_type_ptr=None, size=size)
                    else:
                        h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                        h = h @ weight[i]

                    # 应用关系门控
                    h = self.apply_relation_gate(h, i)

                    relation_features.append(h)

                # 计算注意力权重并聚合（关系间门控）
                if len(relation_features) > 0:
                    stacked_features = torch.stack(relation_features, dim=0)
                    attention_scores = torch.matmul(stacked_features, self.attention)
                    attention_weights = F.softmax(attention_scores, dim=0)
                    out = torch.sum(attention_weights * stacked_features, dim=0)
                    # out = 2*torch.mean(stacked_features, dim=0)

                    # stacked_features= attention_weights * stacked_features
                    # pool_features = stacked_features.mean(dim=1)
                    # attention_scores1=torch.matmul(pool_features, self.attention1)
                    # attention_weights1 = F.softmax(attention_scores1, dim=0)
                    # out = torch.sum(attention_weights1.unsqueeze(1) * stacked_features, dim=0)

        # 保留原有的root weight
        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def forward_single_relation(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                                edge_index: Adj, edge_type: OptTensor = None,
                                target_rel: int = 0, include_root: bool = True) -> Tensor:
        """Forward pass for a single relation type.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_type: Edge type labels
            target_rel: Target relation type to extract features for
            include_root: Whether to include self-loop (root weight).
                          False for pure message aggregation analysis.

        Returns:
            Feature tensor for the specified relation type
        """
        # Convert input features
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        # Filter edges for the target relation
        tmp = masked_edge_index(edge_index, edge_type == target_rel)

        if not torch.is_floating_point(x_r):
            h = self.propagate(tmp, x=weight[target_rel, x_l], edge_type_ptr=None, size=size)
        else:
            h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
            h = h @ weight[target_rel]

        # Apply relation gate
        h = self.apply_relation_gate(h, target_rel)

        # Add root weight only if requested
        if include_root:
            root = self.root
            if root is not None:
                if not torch.is_floating_point(x_r):
                    h = h + root[x_r]
                else:
                    h = h + x_r @ root

        return h

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if (torch_geometric.typing.WITH_SEGMM and not is_compiling()
                and edge_type_ptr is not None):
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations}, '
                f'use_relation_gate={self.use_relation_gate})')