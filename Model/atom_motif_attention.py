import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData, Batch
from typing import Dict, List, Optional, Tuple, Union

"""
    # Create dummy data
    hetero_data = HeteroData()
    
    # Add atom features
    hetero_data['atom'].x = torch.randn(num_atoms, 9)  # 9-dim atom features
    hetero_data['atom'].batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 atoms per molecule
    
    # Add motif features
    hetero_data['motif'].x = torch.randn(num_motifs, 5)  # 5-dim motif features
    hetero_data['motif'].batch = torch.tensor([0, 0, 1, 1])  # 2 motifs per molecule
    
    # Add atom-in-motif edges
    atom_idx = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9])
    motif_idx = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    hetero_data['atom', 'in', 'motif'].edge_index = torch.stack([atom_idx, motif_idx])
    
"""

class AtomMotifAttention(nn.Module):
    def __init__(self, 
                 atom_dim: int,
                 motif_dim: int,
                 output_dim: int = None,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Computes attention between atoms and motifs in molecular graphs.
        Motifs are substructures that contain atoms.
        
        Args:
            atom_dim (int): Dimension of atom node features
            motif_dim (int): Dimension of motif node features
            output_dim (int, optional): Output dimension. Defaults to atom_dim if None.
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(AtomMotifAttention, self).__init__()
        
        self.atom_dim = atom_dim
        self.motif_dim = motif_dim
        self.output_dim = output_dim if output_dim is not None else atom_dim
        self.num_heads = num_heads

        assert self.atom_dim % self.num_heads == 0, "atom_dim must be divisible by num_heads"
        
        self.head_dim = atom_dim // self.num_heads
        
        # 线性变换 - 使用相同维度进行投影
        self.atom_query = nn.Linear(atom_dim, atom_dim)
        self.motif_key = nn.Linear(motif_dim, motif_dim)
        self.motif_value = nn.Linear(motif_dim, motif_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(atom_dim, self.output_dim)
        
        # Dropout防止过拟合
        # self.dropout = nn.Dropout(dropout)
        
        # 层归一化 - 稳定训练
        self.layer_norm = nn.LayerNorm(self.output_dim)


    def forward(self,
                hetero_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算原子与motif之间的注意力，并更新原子特征
        
        Args:
            hetero_data (HeteroData): PyTorch Geometric HeteroData对象，包含:
                - 'atom.x': 原子特征 [num_atoms, atom_dim]
                - 'motif.x': Motif特征 [num_motifs, motif_dim]
                - 'atom.batch': 原子批次分配 [num_atoms]
                - 'motif.batch': Motif批次分配 [num_motifs]
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - 更新后的原子特征 [num_atoms, output_dim]
                - 原子与motif的注意力权重 [num_atoms, num_heads, num_motifs]
        """
        # 提取特征和关系数据
        atom_features = hetero_data['atom'].x  # [num_atoms, atom_dim]
        motif_features = hetero_data['motif'].x  # [num_motifs, motif_dim]

        # 获取批次信息
        atom_batch = hetero_data['atom'].batch  # [num_atoms]
        motif_batch = hetero_data['motif'].batch  # [num_motifs]
        
        # 维度信息
        num_atoms = atom_features.size(0)
        num_motifs = motif_features.size(0)
        device = atom_features.device
        
        # 创建批次掩码，确保原子只关注同一批次中的motif
        batch_mask = atom_batch.view(-1, 1) == motif_batch.view(1, -1)  # [num_atoms, num_motifs]
        batch_mask = batch_mask.to(device)
        
        # 投影特征
        q = self.atom_query(atom_features)  # [num_atoms, atom_dim]
        k = self.motif_key(motif_features)  # [num_motifs, atom_dim]
        v = self.motif_value(motif_features)  # [num_motifs, atom_dim]
        
        # 重塑为多头注意力
        q = q.view(num_atoms, self.num_heads, self.head_dim)  # [num_atoms, num_heads, head_dim]
        k = k.view(num_motifs, self.num_heads, self.head_dim)  # [num_motifs, num_heads, head_dim]
        v = v.view(num_motifs, self.num_heads, self.head_dim)  # [num_motifs, num_heads, head_dim]
        
        # 计算注意力分数: [num_atoms, num_heads, num_motifs]
        scores = torch.einsum('nhd,mhd->nhm', q, k) / (self.head_dim ** 0.5)
        
        # 应用批次掩码
        mask_value = -1e9  # 使用-1e9代替-inf防止梯度问题
        mask_expanded = ~batch_mask.unsqueeze(1)  # [num_atoms, 1, num_motifs]
        scores = scores.masked_fill(mask_expanded, mask_value)
        
        # 应用数值稳定的softmax: 先减去最大值
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(scores, dim=-1)  # [num_atoms, num_heads, num_motifs]
        # attn = self.dropout(attn)
        
        # 计算加权输出
        output = torch.einsum('nhm,mhd->nhd', attn, v)  # [num_atoms, num_heads, head_dim]
        
        # 重塑回原始维度
        output = output.reshape(num_atoms, self.atom_dim)  # [num_atoms, atom_dim]
        
        # 应用输出投影
        output = self.output_proj(output)  # [num_atoms, output_dim]
        
        # 残差连接和层归一化
        if atom_features.size(1) >= self.output_dim:
            # 如果原始特征维度足够大，直接使用
            residual = atom_features[:, :self.output_dim]
        else:
            # 否则，填充到所需维度
            residual = F.pad(atom_features, (0, self.output_dim - atom_features.size(1)))
        
        output = self.layer_norm(output + residual)
        
        return output, attn

    def get_atom_to_atom_attention_efficient(self,
                                           hetero_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        高效计算基于motif的原子间注意力

        Args:
            hetero_data (HeteroData): PyTorch Geometric HeteroData对象

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 更新后的原子特征 [num_atoms, output_dim]
                - 原子间的注意力权重 [num_heads, num_atoms, num_atoms]
        """
        # 获取原子-motif注意力权重
        output, atom_to_motif_attn = self.forward(hetero_data)  # [num_atoms, num_heads, num_motifs]

        # 获取关系信息
        if ('atom', 'in', 'motif') in hetero_data.edge_types:
            edge_index = hetero_data['atom', 'in', 'motif'].edge_index
            atom_indices = edge_index[0]  # 原子索引
            motif_indices = edge_index[1]  # motif索引
        else:
            raise ValueError("无法计算原子间注意力：缺少atom-motif关系")

        # 获取批次信息
        atom_batch = hetero_data['atom'].batch  # [num_atoms]

        # 维度信息
        num_atoms = hetero_data['atom'].x.size(0)
        num_motifs = hetero_data['motif'].x.size(0)
        device = hetero_data['atom'].x.device

        # 创建原子-motif成员关系矩阵（二值化）
        membership = torch.zeros((num_atoms, num_motifs), device=device)
        membership[atom_indices, motif_indices] = 1.0

        # 计算每个原子属于的motif数量
        motifs_per_atom = membership.sum(dim=1)  # [num_atoms]
        # 添加小常数避免除零
        motifs_per_atom = torch.clamp(motifs_per_atom, min=1.0)

        # 归一化成员关系矩阵
        membership_norm = membership / motifs_per_atom.unsqueeze(1)  # [num_atoms, num_motifs]

        # 计算原子间注意力
        # 转置atom_to_motif_attn以确保维度一致
        atom_to_motif_attn_t = atom_to_motif_attn.permute(1, 0, 2)  # [num_heads, num_atoms, num_motifs]

        # 计算原子间注意力：如果原子i关注motif m，而原子j在motif m中，则i间接关注j
        # [num_heads, num_atoms, num_motifs] @ [num_motifs, num_atoms] -> [num_heads, num_atoms, num_atoms]
        atom_to_atom_attn = torch.matmul(atom_to_motif_attn_t, membership_norm.t())

        # 应用批次掩码确保只关注同一图内的原子
        batch_mask = atom_batch.view(1, -1, 1) == atom_batch.view(1, 1, -1)  # [1, num_atoms, num_atoms]
        batch_mask = batch_mask.expand(self.num_heads, -1, -1)  # [num_heads, num_atoms, num_atoms]
        atom_to_atom_attn = atom_to_atom_attn * batch_mask

        # 归一化atom-to-atom注意力
        row_sums = atom_to_atom_attn.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-10)  # 防止除零
        atom_to_atom_attn = atom_to_atom_attn / row_sums

        return output, atom_to_atom_attn


if __name__ == "__main__":
    # This is an example of how to use the model with explicit edges
    import torch_geometric.transforms as T

    print("Example with atom-in-motif edges:")
    num_atoms = 10
    num_motifs = 4

    # Create dummy data
    hetero_data = HeteroData()

    # Add atom features
    hetero_data['atom'].x = torch.randn(num_atoms, 16)  # 9-dim atom features
    hetero_data['atom'].batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 atoms per molecule

    # Add motif features
    hetero_data['motif'].x = torch.randn(num_motifs, 16)  # 5-dim motif features
    hetero_data['motif'].batch = torch.tensor([0, 0, 1, 1])  # 2 motifs per molecule

    # Add atom-in-motif edges
    atom_idx = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9])
    motif_idx = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    hetero_data['atom', 'in', 'motif'].edge_index = torch.stack([atom_idx, motif_idx])

    # Compute atom-to-motif attention

    model = AtomMotifAttention(atom_dim=16, motif_dim=16)

    # Move model to same device as input data
    device = hetero_data['atom'].x.device
    model = model.to(device)

    # Compute atom-to-motif average attention
    updated_atom_features, global_attention_weights = model(hetero_data)
    print(updated_atom_features.shape)
    print(global_attention_weights.shape)

    # Compute atom-to-atom attention
    atom_to_atom_attn = model.get_atom_to_atom_attention_efficient(hetero_data)
    print(atom_to_atom_attn.shape)




