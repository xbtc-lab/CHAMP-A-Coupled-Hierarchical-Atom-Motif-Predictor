import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import MoleculeNet

import os
from rdkit import Chem
from IPython.display import SVG

# 每次运行都重新加载
import importlib
from motif_extract import mol_motif
importlib.reload(mol_motif)

# 将motif转为Data
# rdkit标记的原子序号与torch_geometric中的数据的序号完全一致

import torch
# 给一个motif，得到对于的edge_index(从0开始)和索引(用于找到edge_attr)
def motif_in_edge(data, motif):
    # 提取 motif 中的原子索引
    atom_indices = list(motif)

    # 创建子图的节点特征和边索引
    edge_index = []  # 边索引
    edge_index_indices = []

    # 构建边索引
    for i in range(data.edge_index.size(1)):
        start = data.edge_index[0, i].item()
        end = data.edge_index[1, i].item()
        if start in atom_indices and end in atom_indices:
            # 映射到子图的原子索引
            new_start = atom_indices.index(start)
            new_end = atom_indices.index(end)
            edge_index.append([new_start, new_end])
            edge_index_indices.append(i)

    # 转换为 torch.Tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index_indices = torch.tensor(edge_index_indices, dtype=torch.long)
    # 返回 Data 对象
    return edge_index,edge_index_indices

# ToDo: 保证motif内部的同构性(包含边)
class MotifGINLayer(MessagePassing):
    """同构感知的消息传递层"""

    def __init__(self, hidden_dim):
        super().__init__(aggr="add")  # 使用sum聚合保证WL同构性

        # 学习节点同构性信息
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))  # 可学习的扰动参数

    def forward(self, x, edge_index, edge_attr,is_edge):
        # 消息传播与聚合： （1）message  （2）update  （3）agg
        if is_edge:
            out = self.propagate(
                edge_index,
                x=x,
                edge_attr=edge_attr
            )
            # 中心节点残差连接
            return self.node_mlp((1 + self.eps) * x + out)
        else:
            return self.node_mlp(x)

    def message(self, x_j, edge_attr):
        """消息计算：边特征动态调制节点特征"""
        return edge_attr * x_j  # 逐元素相乘 (E, hidden_dim)

# ToDo: motif信息嵌入:修改学习方法：
class MotifEncoder(nn.Module):
    """完整的Motif编码模型"""
    def __init__(self,
                 # 输入维度
                 atom_feature_dim=9,  # 原子特征维度
                 edge_feat_dim=3,  # 边特征维度
                 # 输出维度：
                 hidden_dim=16,
                 type_hidden_dim=16,
                 num_layers=2):
        super().__init__()
        self.type_hidden_dim = type_hidden_dim
        self.hidden_dim = hidden_dim

        # 0.类型编码函数
        self.type_embedder = torch.nn.Embedding(4, type_hidden_dim)

        # 1. 原子编码函数
        self.atom_nn = torch.nn.Linear(atom_feature_dim, hidden_dim)

        # 2.键编码函数
        self.edge_nn = torch.nn.Linear(edge_feat_dim, hidden_dim)

        # 3.GIN用于学习motif的同构性
        self.convs = nn.ModuleList([
            MotifGINLayer(hidden_dim) for _ in range(num_layers)
        ])

        # 5. 节点注意力
        self.node_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),  # 输入为[节点特征, 边聚合特征]
            nn.Sigmoid()
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),  # 输入为[hi, hj, e_ij]
            nn.ReLU()
        )

        # 6.motif编码
        self.motif_nn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.motif_edge_nn = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, data: Data,motifs_node,motifs_type,motif_edge_index,motif_edge_attr):
        motifs_X = torch.empty(0,self.hidden_dim)
        # 先对每一个motif进行编码
        for type,motif in zip(motifs_type,motifs_node):
            # 1.motif类型编码
            motif_type_node = self.type_embedder(torch.tensor(type))
            motif_type_node[self.type_hidden_dim // 2:] = motif_type_node[self.type_hidden_dim // 2:] * data.x.size(0)

            # 2.GIN保证同构性
            # 1.原子编码
            x = self.atom_nn(torch.index_select(data.x,0,torch.tensor(list(motif))).float())
            # 2. 边的编码
            node_edge_index, node_edge_index_indices = motif_in_edge(data, motif)    # motif内部索引(node_edge_index与motif的原子索引相一致),shape = [2,E]
            # 有边
            if node_edge_index.shape[0] != 0:
                edge_attr = torch.index_select(data.edge_attr, 0, node_edge_index_indices).float()
                edge_attr = self.edge_nn(edge_attr)    # (E, edge_feat_dim)
                for conv in self.convs:
                    x = conv(x, node_edge_index, edge_attr, True)

                edge_agg = torch.zeros_like(x)
                row, col = node_edge_index
                # 根据边的类型计算权重
                for i in range(edge_attr.size(0)):
                    edge_agg[row[i]] += edge_attr[i, :]
                    edge_agg[col[i]] += edge_attr[i, :]

                # 节点级注意力: 这里的edge_agg不太对劲
                alpha = self.node_attn(torch.cat([x, edge_agg], dim=-1))  # (N,1)
                h_nodes = torch.sum(alpha * x, dim=0)  # (hidden_dim)

                # 边-节点联合编码:
                h_edges = []
                for i in range(node_edge_index.size(1)):
                    src, dst = node_edge_index[:, i]
                    h_edge = self.edge_mlp(
                        torch.cat([x[src], x[dst], edge_attr[i]], dim=-1)
                    )
                    h_edges.append(h_edge)

                h_edges = torch.mean(torch.stack(h_edges), dim=0)  # (hidden_dim)
                motif_embedding = self.motif_nn(torch.concat([h_nodes + h_edges, motif_type_node], dim=-1)).unsqueeze(0)  # (hidden_dim)
            # 无边
            else:
                # 同构性编码
                for conv in self.convs:
                    x = conv(x, None, None, False)
                # 读出
                h_nodes = x.squeeze(0)
                motif_embedding = self.motif_nn(torch.concat([h_nodes, motif_type_node], dim=-1)).unsqueeze(0)  # (hidden_dim)

            motifs_X = torch.cat((motifs_X, motif_embedding), dim=0)

        # motif边进行编码：两边的motif类型编码 + 节点和边的编码 motif_edge_index.shape = [2,E],motif_edge_attr.shape = [E,]:
        motifs_edge_attr = torch.empty(0,self.hidden_dim)
        for i,(attr) in enumerate(motif_edge_attr):
            # 找他们之间motif的类型
            start = motifs_type[motif_edge_index[:,i][0].item()]
            end   = motifs_type[motif_edge_index[:,i][1].item()]
            motif_type_edge = (self.type_embedder(torch.tensor(start)) + self.type_embedder(torch.tensor(end))).unsqueeze(0)
            # 保持原边和节点:两个节点夹住中间的边
            h_atom_edge = torch.zeros(1, self.hidden_dim)
            h_atom_edge += self.atom_nn(data.x[attr["node"],:].float()).sum(dim=0)
            h_atom_edge += self.edge_nn(data.edge_attr[attr["edge"],:].float()).sum(dim=0)
            # 合并
            motif_attr_embedding = self.motif_edge_nn(torch.cat((motif_type_edge, h_atom_edge), dim=-1))
            motifs_edge_attr = torch.cat((motifs_edge_attr, motif_attr_embedding), dim=0)

        # 合并他们
        motif_Data = Data(x = motifs_X,edge_index=motif_edge_index,edge_attr = motifs_edge_attr,smiles = data.smiles,y=data.y)
        return motif_Data

# 找到node_list之间的全部连接边
def find_unique_edges_with_indices(node_list, edge_index):
    """
    找到给定节点列表中节点之间是否存在连接的边，并去重（支持 PyTorch Tensor）。
    同时返回每条边在 edge_index 中的索引。

    参数:
        node_list (list): 给定的节点列表。
        edge_index (torch.Tensor): 图的边索引，形状为 [2, num_edges]。

    返回:
        tuple:
            - list of tuples: 包含所有唯一连接的边的列表，格式为 [(src, dst), ...]。
            - list of int: 每条唯一边在 edge_index 中的索引。
    """
    # 确保 edge_index 是一个二维张量，形状为 [2, num_edges]
    if edge_index.size(0) != 2:
        raise ValueError("edge_index 必须是形状为 [2, num_edges] 的张量")

    # 创建节点集合以便快速查找
    node_set = set(node_list)

    # 提取源节点和目标节点
    src_nodes = edge_index[0]  # 第一行：源节点
    dst_nodes = edge_index[1]  # 第二行：目标节点

    # 筛选出节点列表中的节点之间的边，并标准化为 (min(src, dst), max(src, dst))
    unique_edges = set()
    edge_indices = []  # 记录符合条件的边的索引

    for i, (src, dst) in enumerate(zip(src_nodes.tolist(), dst_nodes.tolist())):
        if src in node_set and dst in node_set:
            normalized_edge = tuple(sorted((src, dst)))  # 标准化边
            if normalized_edge not in unique_edges:
                unique_edges.add(normalized_edge)
                edge_indices.append(i)  # 记录该边的索引

    # 转换回列表并返回
    return list(unique_edges), edge_indices[0]


# 找到motif之间的边和节点:得到的motif的edge_index的索引与motif的索引一致
def get_motif_edge(data, motifs_result):
    """
    构建 motif-level 图并返回 torch_geometric.data.Data 格式的图。
    都是以单键的形式存在的。这些单链只作连接的作用
    1.无交集：看motif内部 ：边
    2.有两个交集：环类 ：边
    3.一个交集：怎么办：点

    参数:
        smiles (str): 分子的 SMILES 字符串。
        motifs_result (list of set): 每个 motif 的节点集合。
        motif_X (list of tensor): 每个 motif 的嵌入信息。

    返回:
        torch_geometric.data.Data: 包含 motif-level 图的拓扑结构、节点信息和边信息。
    """
    # 1. 解析 SMILES 获取分子的拓扑结构
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # 获取原子数和键信息
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

    # 2. 初始化 motif-level 图的节点和边
    num_motifs = len(motifs_result)
    edge_index = []  # 存储 motif-level 图的边
    edge_attr = []  # 存储边的属性（保留原 SMILES 中的边信息）
    edge_attr_dim = data.edge_attr.shape[1]
    # 将 motif 转换为列表以便索引
    motifs_result = [list(motif) for motif in motifs_result]

    # 3. 构建 motif-level 图的边
    for i in range(num_motifs):
        for j in range(i + 1, num_motifs):

            # 检查 motif_i 和 motif_j 是否有交集
            intersection = set(motifs_result[i]).intersection(set(motifs_result[j]))

            # 如果有交集，直接连接 motif_i 和 motif_j 并获取交集的信息
            if intersection:

                # 如果有大于2个交集:跳过无效
                if len(intersection) > 2:
                    return None

                edge_index.append([j, i])
                edge_index.append([i, j])

                # 如果有1个交集
                if len(intersection) == 1:
                    atom = next(iter(intersection))
                    edge_attr.append([0] * edge_attr_dim + [atom,atom])      # 边 点 点
                    edge_attr.append([0] * edge_attr_dim + [atom,atom])      # 边 点 点

                # 如果有2个交集
                elif len(intersection) == 2:
                    # 对于有交集的情况，返回他们的交集以及在edge_index的索引
                    edge,edge_indices = find_unique_edges_with_indices(list(intersection),data.edge_index)
                    edge_attr.append([k.item() for k in data.edge_attr[edge_indices]] + list(intersection))
                    edge_attr.append([k.item() for k in data.edge_attr[edge_indices]] + list(intersection))


            # 如果他们内部
            # 如果没有交集，检查 motif 内部的节点边是否连接两个 motif
            else:
                for u in motifs_result[i]:
                    for v in motifs_result[j]:
                        if (u, v) in bonds or (v, u) in bonds:
                            # 如果 motif 内部的节点边连接两个 motif，添加边
                            edge_index.append([i, j])
                            edge_index.append([j, i])  # 无向图，添加反向边
                            # 找到索引：
                            index = [k for k in range(data.edge_index.size(1)) if data.edge_index[0, k].item() == u and data.edge_index[1, k].item() == v][0]

                            edge_attr.append([k.item() for k in data.edge_attr[index]] + [u,v])
                            edge_attr.append([k.item() for k in data.edge_attr[index]] + [u,v])
                            break

    # 5. 构建边索引和边属性
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # 转置为 [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)  # 边属性

    return edge_index,edge_attr

if __name__ == '__main__':
    # 数据集
    datasets = MoleculeNet(root="./dataset/", name="Lipo")
    Data_motif_list = []
    # motif_graph = MotifEncoder(
    #              # 输入维度
    #              atom_feature_dim=9,  # 原子特征维度
    #              edge_feat_dim=3,  # 边特征维度
    #              # 输出维度：
    #              hidden_dim=16,
    #              type_hidden_dim=16,
    #              num_layers=2)


    for i,data in enumerate(datasets[:100]):
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        print(smiles)
        # print(i, smiles)
        if mol is not None:
            try:
                # 划分子结构 以及 子结构类型
                motifs_type,motifs_node = mol_motif.mol_get_motif(mol)

                # 找到motif的边：
                if get_motif_edge(data, motifs_node) is None:
                    continue
                motif_edge_index,motif_edge_attr = get_motif_edge(data, motifs_node)

                motifs_type = torch.tensor(motifs_type, dtype=torch.long)

                print(motifs_node)

                # 保存图片
                svg = mol_motif.visualize_motif(mol, motifs_node,method='save')
                with open(f'./dataset/Lipo_image/{i}.png', 'wb') as file:
                    file.write(svg)
                    print(f"第{i}张保存完毕")

                # 开始训练：

                # motif_X = []
                # for motif in motifs_node:
                #     # 得到motif_Data
                #     motif_type = next(iter([k for k, lst in motifs_type.items() if motif in lst]))
                #     motifs_Data = motif_to_Data(data, motif, motif_type)
                #     # 得到motif的嵌入 和 h_atom
                #     h_atom, h_motif = motif_encoder(motifs_Data)
                #     # 将h_atom嵌入到data中
                #     motif_X.append(h_motif)
                # # 需要做的是：给定官能团序列，官能团类型序列，他们之间贡献的边和原子 -》得到motif_graph
                # # 给定
                # Data_motif = get_motif_edge(data, motifs_node, motif_X)
                # Data_motif_list.append(Data_motif)

            except Exception as e:
                print(print(f"Error processing SMILES '{i}:{data.smiles}': {e}"))


    # 计算
    # dataset = GraphDataset(root='./dataset/motif_Tox21', data_list=Data_motif_list)

    # 对于输入的每一个分子
    """
    (1)分子-》motif集合
    (2)学习每个motif的嵌入，并且保存atom的嵌入
    (2)给定motif集合得到motif_level的图:图中要有
    """

    # 对于每一个分子，都应该保存：atom_Data,motif_Data


    # motifs_X =





