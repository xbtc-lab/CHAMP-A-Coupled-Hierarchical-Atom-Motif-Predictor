import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HMSAF(nn.Module):
    """
    Enhanced DCMHAttention that implements:
    1. Basic attention (A_ij)
    2. Head interaction matrix (Wb) for cross-head information flow
    3. Dynamic gating
    """
    def __init__(self,
                 n_head,
                 input_dim,
                 output_dim,
                 use_Guide=True,
                 use_gating=True,
                 use_head_interaction=True,
                 dropout=0.1):

        super().__init__()
        # Define dimensions
        assert output_dim % n_head == 0

        self.n_head = n_head
        self.output_dim = output_dim
        self.input_dim = input_dim
        # 修复head_dim的计算 - 应该是output_dim // n_head而非input_dim // n_head
        self.head_dim = output_dim // n_head
        self.use_head_interaction = use_head_interaction
        self.use_gating = use_gating
        self.use_Guide = use_Guide
        # self.dropout = nn.Dropout(dropout)

        # Key, query, value projections
        self.wq = nn.Linear(input_dim, output_dim, bias=True)
        self.wk = nn.Linear(input_dim, output_dim, bias=True)
        self.wv = nn.Linear(input_dim, output_dim, bias=True)
        self.wo = nn.Linear(output_dim, input_dim, bias=True)

        # 添加层归一化
        self.layer_norm = nn.LayerNorm(input_dim)

        # 头交互矩阵 (Wb) - 形状为 [n_head, n_head]
        if self.use_Guide:
            # atom_motif_attn 与base_attn的结合
            self.attn_cross_particle = nn.Parameter(torch.randn(n_head, n_head))

            # 加权参数 - 使用较小的初始值
            self.alpha = nn.Parameter(torch.tensor(0.5))  # attn_scores 的权重
            self.beta = nn.Parameter(torch.tensor(0.5))  # motif_to_atom_attn 结果的权重

        # 动态门控组件 (Dynamic Gating)
        if self.use_gating:
            # self.gate_proj_k_cross = nn.Parameter(torch.randn(n_head, n_head))
            self.gate_proj_k = nn.Linear(output_dim, n_head, bias=True)

            # self.gate_proj_q_cross = nn.Parameter(torch.randn(n_head, n_head))
            self.gate_proj_q = nn.Linear(output_dim, n_head, bias=True)

        if self.use_head_interaction:
            self.head_interaction = nn.Parameter(
                torch.eye(n_head) + 0.01 * torch.randn(n_head, n_head)
            )

        self.scale_factor = 1 / math.sqrt(self.head_dim)

    # 原子结果 + 原子与motif    1.融合    2.使用动态调整  3.头交互
    def forward(self, x: torch.Tensor, batch, motif_to_atom_attn) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入节点特征 [num_nodes, input_dim]
            batch: 节点所属批次 [num_nodes]
            motif_to_atom_attn: motif到atom的注意力矩阵 [n_head, num_nodes, num_nodes]

        Returns:
            torch.Tensor: 更新后的节点特征 [num_nodes, input_dim]
        """
        num_atom, input_dim = x.shape
        identity = x  # 保存用于残差连接

        # 将输入投影到查询、键、值空间
        q = self.wq(x).view(num_atom, self.n_head, self.head_dim)
        k = self.wk(x).view(num_atom, self.n_head, self.head_dim)
        v = self.wv(x).view(num_atom, self.n_head, self.head_dim)

        # 转置为 [n_head, num_atom, head_dim]
        q = q.transpose(0, 1)  # [H, N, d]
        k = k.transpose(0, 1)  # [H, N, d]
        v = v.transpose(0, 1)  # [H, N, d]

        # 计算注意力分数 (A_ij) - 基础注意力 [H, N, N]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor

        # 应用批处理掩码，确保只关注同一图中的节点
        if batch is not None:
            batch_mask = torch.eq(batch.unsqueeze(-1), batch.unsqueeze(-2))  # [N, N]
            batch_mask = batch_mask.unsqueeze(0).expand(self.n_head, -1, -1)  # [H, N, N]
            # 使用-1e9代替-inf以避免梯度问题
            attn_scores = attn_scores.masked_fill(~batch_mask, -1e9)

        attn_final = attn_scores                    #  [H, N, N]
        attn_scores = attn_scores.permute(1, 2, 0)  # [N, N, H]

        # 应用头交互矩阵 (Wb)
        if self.use_Guide:
            # 将motif_to_atom_attn转换为[N, N, H]格式
            motif_to_atom_attn = motif_to_atom_attn.permute(1, 2, 0)

            # 加权融合两种注意力
            # attn_base = self.alpha * attn_scores + self.beta * torch.einsum('stn,nm->stm', motif_to_atom_attn,
            #                                                                 self.attn_cross_particle)
            attn_base = self.alpha * attn_scores + self.beta * motif_to_atom_attn
            # ============================================================================：修改前
            # # 应用头交互：[N, N, H] @ [H, H] -> [N, N, H]
            # attn_base_interaction = torch.einsum('stn,nm->stm', attn_base, self.head_interaction)
            # # 转回 [H, N, N] 格式
            # attn_base_interaction = attn_base_interaction.permute(2, 0, 1)

            # ============================================================================:修改后
            attn_base_interaction = attn_base.permute(2, 1, 0)

            attn_final = attn_final + attn_base_interaction


        # 应用动态门控
        if self.use_gating:
            # 获取扁平化的查询和键表示
            q_flat = q.transpose(0, 1).reshape(num_atom, -1)  # [N, H*d]
            k_flat = k.transpose(0, 1).reshape(num_atom, -1)  # [N, H*d]

            # 生成门控值
            gates_q = self.gate_proj_q(q_flat)  # [N, H]
            gates_q = torch.tanh(gates_q)  # [N, H]

            gates_k = self.gate_proj_k(k_flat)  # [N, H]
            gates_k = torch.tanh(gates_k)  # [N, H]

            # 调整门控维度以匹配注意力分数: 用于表示
            gates_q = gates_q.transpose(0, 1).unsqueeze(-1)  # [H, N, 1]
            gates_k = gates_k.transpose(0, 1).unsqueeze(1)  # [H, 1, N]

            # 获取原始注意力分数
            raw_attn = attn_scores.permute(2, 0, 1)  # [H, N, N]

            # 应用门控
            gated_attn = raw_attn * gates_q + raw_attn * gates_k  # [H, N, N]
            attn_final = attn_final + gated_attn


        if self.use_head_interaction:
            attn_head_intercation = torch.einsum('stn,nm->stm', attn_scores, self.head_interaction).permute(2, 0, 1)
            attn_final = attn_final + attn_head_intercation



        # 使用数值稳定的softmax：先减去最大值
        # attn_final = attn_final - attn_final.max(dim=-1, keepdim=True)[0]   # 注释掉

        # 应用softmax获取注意力概率
        attn_probs = F.softmax(attn_final, dim=-1)  # [H, N, N]

        # 应用dropout
        # attn_probs = self.dropout(attn_probs)

        # 将注意力应用到值
        output = torch.matmul(attn_probs, v)  # [H, N, d]

        # 重塑输出 [H, N, d] -> [N, H*d]
        output = output.transpose(0, 1).contiguous().view(num_atom, -1)

        # 最终投影
        output = self.wo(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + identity)

        return output,attn_final
        # return output

