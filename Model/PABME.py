import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv, GINEConv,EGConv,FAConv,FiLMConv,PANConv,PNAConv

class EdgeMLP(torch.nn.Module):
    def __init__(self,hidden_dim):
        super(EdgeMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim)
        )
    def forward(self, edge_attr):
        return self.mlp(edge_attr)


class GINLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GINLayer, self).__init__()
        nn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINConv(nn_mlp)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)

class GINELayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GINELayer, self).__init__()
        nn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(nn_mlp)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


# TODO：用于信息传递
class CustomGNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super(CustomGNNLayer, self).__init__(aggr='add')  # 使用加法聚合
        self.node_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim)
        )
        self.edge_mlp = EdgeMLP(hidden_dim)
        self.epsilon = torch.nn.Parameter(torch.tensor([0.]))  # 可学习的参数 ε

    def forward(self, x, edge_index, edge_attr):
        # 调用消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # 更新节点表示
        out = self.node_mlp(out)
        return out

    def message(self, x_j, edge_attr):
        # 边特征处理
        edge_features = self.edge_mlp(edge_attr)
        # 邻接节点的消息
        return edge_features * x_j

    def update(self, aggr_out, x):
        # 自环信息
        self_eps = (1 + self.epsilon) * x
        # 聚合后的邻居信息
        return aggr_out + self_eps



class GCNLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)

class GATLayer(torch.nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super(GATLayer, self).__init__()
        self.conv = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)



#TODO：用于读出
class HierarchicalEdgePooling(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, Pair_MLP):
        super().__init__()

        # 节点注意力
        self.node_attn_fc = nn.Linear(node_dim + edge_dim, 1)

        # Pair-MLP 注意力
        self.pair_attn_fc = nn.Linear(2 * node_dim + edge_dim, 1)

        # Pair-MLP 编码
        self.Pair_MLP = Pair_MLP
        self.pair_mlp_sum = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)  # 输出与节点特征相同维度
        )

        self.pair_mlp_concat = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)  # 输出与节点特征相同维度
        )

        self.beta = nn.Parameter(torch.tensor(1.0))  # 可学习参数 β

    def forward(self, x, edge_index, edge_attr, motif_atom_edges):
        """
        x: (num_nodes, node_dim) -> 节点特征
        edge_index: (2, num_edges) -> 边索引
        edge_attr: (num_edges, edge_dim) -> 边特征
        motif_atom_edges: (2, num_motif_atom_edges) -> [motif_idx, atom_idx] 对应关系
        """
        # 解构motif-atom对应关系
        motif_idx, atom_idx = motif_atom_edges

        # 为每个motif-atom对创建虚拟节点副本
        x_motif_atom = torch.index_select(x, 0, atom_idx)

        # 现在需要创建新的边索引映射，因为节点索引已经改变
        # 我们需要构建一个映射：(原始atom索引, motif索引) -> 新的虚拟节点索引
        #===================================================================================================
        # 首先创建唯一的(motif_idx, atom_idx)对并为每个对分配一个新索引,并映射字典: (atom_idx, motif_idx) -> new_idx
        # node_pairs = torch.stack([motif_idx, atom_idx], dim=1)
        # unique_pairs, inverse_indices = torch.unique(node_pairs, dim=0, return_inverse=True)
        # mapping = {(row[1].item(), row[0].item()): i for i, row in enumerate(unique_pairs)}
        #
        # # 重新映射原始edge_index到新的虚拟节点索引：设置为一个函数，用于聚合
        # new_edge_index = torch.zeros_like(edge_index)
        # for i in range(edge_index.size(1)):
        #     #
        #     u, v = edge_index[0, i].item(), edge_index[1, i].item()
        #
        #     # search motif_id of src_node and dis_node
        #     motifs_u = motif_idx[atom_idx == u]
        #     motifs_v = motif_idx[atom_idx == v]
        #
        #     # 找到u和v共同所在的motif（如果有）
        #     common_motifs = torch.tensor([m.item() for m in motifs_u if m in motifs_v])
        #
        #     # 肯定属于同一个motif中
        #     if len(common_motifs) > 0:
        #         # 使用第一个公共motif
        #         motif = common_motifs[0].item()
        #         new_u = mapping.get((u, motif), u)
        #         new_v = mapping.get((v, motif), v)
        #         new_edge_index[0, i] = new_u
        #         new_edge_index[1, i] = new_v

        #===================================================================================================
        node_pairs = torch.stack([motif_idx, atom_idx], dim=1)
        unique_pairs, inverse_indices = torch.unique(node_pairs, dim=0, return_inverse=True)
        motif_unique = unique_pairs[:, 0]  # 唯一 motif 下标
        atom_unique = unique_pairs[:, 1]  # 唯一 atom 下标

        # 构建映射表: (motif, atom) -> 新节点索引
        num_motifs = int(motif_idx.max().item()) + 1
        num_nodes = x.size(0)
        mapping = torch.full((num_motifs, num_nodes), -1, dtype=torch.long, device=x.device)
        mapping[motif_unique, atom_unique] = torch.arange(unique_pairs.size(0), device=x.device)

        # 原始边索引拆解
        u, v = edge_index

        # 构造每条边在各 motif 下的 membership mask
        membership = mapping >= 0  # (num_motifs, num_nodes)
        mask_u = membership[:, u]  # (num_motifs, num_edges)
        mask_v = membership[:, v]
        mask_common = mask_u & mask_v  # 两端原子都属于该 motif

        # 判断哪些边有公共 motif，并取第一个公共 motif
        has_common = mask_common.any(dim=0)  # (num_edges,)
        first_common_motif = mask_common.float().argmax(dim=0).long()  # (num_edges,)

        # 根据公共 motif 和映射表重新索引
        new_u = torch.where(has_common,
                            mapping[first_common_motif, u],
                            u)
        new_v = torch.where(has_common,
                            mapping[first_common_motif, v],
                            v)

        new_edge_index = torch.stack([new_u, new_v], dim=0)
        row, col = new_edge_index

        #===================================================================================================
        # 计算邻居边特征的加和 ∑ e_uv
        e_sum = torch.zeros(size=(x_motif_atom.size(0), edge_attr.size(1)), device=x.device)
        e_sum.index_add_(0, col, edge_attr)  # 将每个节点的入边特征相加

        # 计算注意力权重 α_v：node_alpha
        node_attn_input = torch.cat([x_motif_atom, e_sum], dim=1)  # [h_v; ∑ e_uv]
        node_alpha = torch.sigmoid(self.node_attn_fc(node_attn_input))                  # node_alpha:[N,1]
        h_pool = global_add_pool(node_alpha * x_motif_atom, motif_idx)


        # 计算 Pair-MLP 编码


        # 对于边池化，我们需要找到每条边所属的motif:new_edge_index是新的
        #========================================================================
        # edge_motif = torch.zeros_like(row)
        # for i in range(row.size(0)):
        #     # 获取边的两个端点
        #     u, v = row[i].item(), col[i].item()
        #
        #     # 找到这两个atom共同所在的motif
        #     motif = motif_idx[inverse_indices[u]]  # 可以用任一端点的motif
        #     edge_motif[i] = motif

        #========================================================================
        node_indices = inverse_indices[row]  # shape: (num_edges,)
        edge_motif = motif_idx[node_indices]  # shape: (num_edges,)

        # h_G = ∑ α_v h_v + β * (1/|E| ∑ Pair-MLP)
        if self.Pair_MLP == True:
            h_u = x_motif_atom[row]
            h_v = x_motif_atom[col]

            pair_input = torch.cat([h_u, h_v, edge_attr], dim=1)  # [h_u; h_v; e_uv]
            pair_alpha = torch.sigmoid(self.pair_attn_fc(pair_input))  # pair_alpha:[E,1]
            pair_output = self.pair_mlp_concat(pair_input)
            pair_output = pair_alpha * pair_output

            edge_pool = global_add_pool(pair_output, edge_motif)
            # 末尾补充:
            padding_size = h_pool.size(0) - edge_pool.size(0)
            padding_pool = torch.zeros(size = (padding_size,edge_pool.size(1)), dtype=edge_pool.dtype, device=edge_pool.device)
            edge_pool = torch.cat([edge_pool, padding_pool], dim=0)

            # 考虑不同motif可能有不同数量的边
            edge_counts = torch.bincount(edge_motif)
            edge_counts = torch.cat([edge_counts,torch.zeros(padding_size,device=edge_counts.device)],dim=0)
            # 防止除零
            edge_counts = torch.clamp(edge_counts, min=1)
            # 扩展为与h_pool相同的形状
            expanded_edge_counts = edge_counts.unsqueeze(1).expand(-1, pair_output.size(1))
            # 正则化edge_pool
            edge_pool = edge_pool / expanded_edge_counts.float()
            h_G = h_pool + self.beta * edge_pool
            return node_alpha, pair_alpha, h_G

        else:
            h_G = h_pool
            pair_alpha = 0
            return node_alpha,pair_alpha,h_G

class GNNModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, Pair_MLP=True,gnn_type="our"):
        super(GNNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.gnn_type = gnn_type
        self.Pair_MLP = Pair_MLP

        for _ in range(num_layers):
            if gnn_type == 'our':
                self.layers.append(CustomGNNLayer(hidden_dim))
            elif gnn_type == 'GCN':
                self.layers.append(GCNLayer(hidden_dim))
            elif gnn_type == 'GAT':
                self.layers.append(GATLayer(hidden_dim, heads=1))
            elif gnn_type == 'GIN':
                self.layers.append(GINLayer(hidden_dim))
            elif gnn_type == 'GINE':
                self.layers.append(GINELayer(hidden_dim))

            # ---- 新增的卷积层 ----
            elif gnn_type == 'EGConv':
                # EGConv requires num_bases and num_heads. Using example values.
                self.layers.append(EGConv(hidden_dim, hidden_dim, num_bases=4, num_heads=4))
            elif gnn_type == 'FAConv':
                self.layers.append(FAConv(hidden_dim, hidden_dim))
            elif gnn_type == 'FiLMConv':
                self.layers.append(FiLMConv(hidden_dim, hidden_dim))
            elif gnn_type == 'PANConv':
                # PANConv does not require num_nodes at init, it can infer it from 'x' in forward
                self.layers.append(PANConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")

        self.final_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.Pool = HierarchicalEdgePooling(hidden_dim, hidden_dim, hidden_dim,self.Pair_MLP)

    def forward(self, x, edge_index, edge_attr, motif_atom_edge_index):
        """
        (1)官能团内部节点的信息传递
        (2)官能团的读出
        (3)官能团编码
        """
        # (1)官能团内部节点的信息传递
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = self.final_mlp(x)

        # (2)官能团的读出:
        node_alpha,pair_alpha,h_g = self.Pool(x, edge_index, edge_attr, motif_atom_edge_index)

        return node_alpha,pair_alpha,h_g, x
        # return h_g, x


