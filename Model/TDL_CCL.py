import torch
from torch_scatter import scatter_add
import torch.nn.functional as F
def compute_ring_contrastive_loss(batch, temperature=0.1, eps=1e-8):
    """
    只对 ring5 和 ring6 的 motif 进行结构感知对比学习损失。
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch]  # [N]
    type_list = batch["motif"].type  # [N]

    # 每个 motif 的原子数量
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)  # [N]

    # 构造 motif_type
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]   # domain
    type_class = [type_prefix[int(t)] for t in type_list]                                                # type

    # 只保留 ring5 和 ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = [m in allowed_types for m in motif_type]
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=z.device)

    # 如果没有足够样本，直接返回 0
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # 过滤有效样本
    z = z[valid_mask]
    labels = labels[valid_mask]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_mask[i]]

    # 转张量
    labels = labels.view(-1, 1)
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)
    type_class_tensor = torch.tensor([hash(t) for t in type_class], device=z.device).view(-1, 1)

    # 相似度矩阵
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # 掩码
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    label_eq = (labels == labels.T)
    label_neq = (labels != labels.T)
    type_eq = (motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1))
    class_eq = (type_class_tensor == type_class_tensor.T)

    pos_mask = label_eq & type_eq & diag_mask     # y同，且domain相同
    neg_mask = label_neq & class_eq & diag_mask   # y不同，并且都是1

    # 对比损失
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    # 正样本对
    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    # 负样本对
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()


import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

def compute_ring_contrastive_loss_multilabel(batch, temperature=0.1, eps=1e-8):
    """
    多标签版本结构感知对比学习损失函数，支持标签中包含 NaN。
    仅对 ring5 和 ring6 motif 进行 contrastive loss 计算。
    """
    z = batch["motif"].x  # [N, D]
    raw_labels = batch["mol"].y[batch["motif"].batch]  # [N, C]
    type_list = batch["motif"].type  # [N]

    # 每个 motif 的原子数量
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)

    # 构造 motif_type
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]
    type_class = [type_prefix[int(t)] for t in type_list]

    # 只保留 ring5 和 ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = torch.tensor([m in allowed_types for m in motif_type], dtype=torch.bool, device=z.device)

    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[valid_mask]
    raw_labels = raw_labels[valid_mask]      # [N, C]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_mask[i]]

    # ========================
    # 过滤标签全为 NaN 的样本
    # ========================
    not_nan_mask = ~torch.isnan(raw_labels)  # [N, C]
    valid_label_mask = not_nan_mask.any(dim=1)  # 至少有一个非NaN标签

    if valid_label_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[valid_label_mask]
    labels = raw_labels[valid_label_mask]
    motif_type = [m for i, m in enumerate(motif_type) if valid_label_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_label_mask[i]]

    # 将 NaN 替换为 0（不会影响 overlap 判断，因为我们只关心是否有交集）
    labels = torch.nan_to_num(labels, nan=0.0)

    # 哈希类别
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)
    type_class_tensor = torch.tensor([hash(t) for t in type_class], device=z.device).view(-1, 1)

    # 相似度
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # 多标签相似性判断
    label_overlap = torch.matmul(labels.float(), labels.T.float())  # [N, N]
    label_eq = label_overlap > 0
    label_neq = label_overlap == 0

    type_eq = motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1)
    class_eq = type_class_tensor == type_class_tensor.T

    pos_mask = label_eq & type_eq & diag_mask
    neg_mask = label_neq & class_eq & diag_mask

    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()





def  compute_nonring_contrastive_loss(batch, threshold=0.9, temperature=0.1, eps=1e-8):
    """
        相同domain且相同label的为正样本
        相同domain但不同label的为负样本

    只对非环（type != 0）motif 执行结构感知对比学习损失。
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch]  # [N]
    type_list = batch["motif"].type  # [N]
    vectors = batch["motif"].vector  # [N, V]

    # 选择非环 motif
    nonring_mask = (type_list != 0)
    if nonring_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # 过滤
    z = z[nonring_mask]
    labels = labels[nonring_mask]
    vectors = vectors[nonring_mask]

    # 相似度矩阵（基于 vector）
    vec_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    domain_mask = (vec_sim_matrix >= threshold)

    # 相似度矩阵（用于对比）
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # 构造 mask
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    label_eq = (labels.view(-1, 1) == labels.view(1, -1))
    label_neq = ~label_eq

    # 正样本：同一 label 且在同一个 domain
    pos_mask = label_eq & domain_mask & diag_mask
    # 负样本：不同 label 但在同一个 domain
    neg_mask = label_neq & domain_mask & diag_mask

    # 如果正样本太少
    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # InfoNCE
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()


def compute_nonring_contrastive_loss_multilabel(batch, threshold=0.9, label_sim_threshold=0.5, temperature=0.1, eps=1e-8):
    """
    多标签 + 缺失值版本的非环 motif 对比损失：
    - 同 domain 且标签相似度 >= label_sim_threshold 为正样本
    - 同 domain 且标签相似度 <  label_sim_threshold 为负样本
    - 忽略标签中有 NaN 的样本
    """

    z = batch["motif"].x  # [N, D]
    labels_raw = batch["mol"].y[batch["motif"].batch]  # [N, C] 多标签
    type_list = batch["motif"].type  # [N]
    vectors = batch["motif"].vector  # [N, V]

    # 选择非环 motif
    nonring_mask = (type_list != 0)
    if nonring_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # 过滤非环
    z = z[nonring_mask]
    labels_raw = labels_raw[nonring_mask]
    vectors = vectors[nonring_mask]

    # 去除含NaN的标签样本
    valid_label_mask = ~torch.isnan(labels_raw).any(dim=1)
    if valid_label_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[valid_label_mask]
    labels = labels_raw[valid_label_mask]  # [N, C]
    vectors = vectors[valid_label_mask]

    # 相似度矩阵（domain）
    vec_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    domain_mask = (vec_sim_matrix >= threshold)

    # 特征对比相似度
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # 标签相似度（多标签 cosine）
    label_sim_matrix = F.cosine_similarity(labels.unsqueeze(1), labels.unsqueeze(0), dim=-1)
    label_eq = label_sim_matrix >= label_sim_threshold
    label_neq = ~label_eq

    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # 正负样本
    pos_mask = label_eq & domain_mask & diag_mask
    neg_mask = label_neq & domain_mask & diag_mask

    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # InfoNCE
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()

def compute_ring_contrastive_loss_regression(batch, temperature=0.1, sigma=1.0, label_thresh_ratio=0.1, eps=1e-8):
    """
    回归任务下，针对 ring5 和 ring6 的 motif 执行结构感知对比学习损失。
    """
    z = batch["motif"].x
    labels = batch["mol"].y[batch["motif"].batch]
    type_list = batch["motif"].type

    # 获取 motif 中原子数量（用作 motif_type 的后缀）
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)

    # 构造 motif_type（domain）
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]

    # 只保留 ring5 和 ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = [m in allowed_types for m in motif_type]
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=z.device)

    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # 过滤有效样本
    z = z[valid_mask]
    labels = labels[valid_mask].view(-1, 1)  # [N, 1]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]

    # 转张量
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)

    # 相似度矩阵
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # [N, N]

    # domain 相同 & 非自身
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    domain_mask = motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1)
    domain_mask = domain_mask & diag_mask

    # 标签差异比例（百分比）
    label_diff = torch.abs(labels - labels.T)
    label_base = torch.max(labels, labels.T) + eps
    relative_diff = label_diff / label_base  # 百分比差异

    # 正样本：domain 相同 且 标签差异小于比例
    pos_mask = domain_mask & (relative_diff < label_thresh_ratio)
    neg_mask = domain_mask & (relative_diff >= label_thresh_ratio)

    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # 标签差异高斯核加权（仅对正样本）
    weight_matrix = torch.exp(- (label_diff ** 2) / (2 * sigma ** 2))
    pos_weight = weight_matrix.masked_fill(~pos_mask, 0.0)

    # 相似度填充
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = (torch.exp(pos_sim / temperature) * pos_weight).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()

def compute_nonring_contrastive_loss_regression(batch, threshold=0.9, label_thresh_ratio=0.1, sigma=1.0, temperature=0.1, eps=1e-8):
    """
    回归任务中，针对非环结构的 motif 进行结构感知对比学习损失。
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch].view(-1, 1)  # [N, 1]
    type_list = batch["motif"].type  # [N]
    vectors = batch["motif"].vector  # [N, V]

    # 选择非环 motif（type != 0）
    nonring_mask = (type_list != 0)
    if nonring_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[nonring_mask]
    labels = labels[nonring_mask]
    vectors = vectors[nonring_mask]

    # 相似度矩阵：基于原子种类向量（定义 domain）
    vec_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    domain_mask = (vec_sim_matrix >= threshold)

    # 标签差值百分比
    label_diff = torch.abs(labels - labels.T)
    label_base = torch.max(labels, labels.T) + eps
    relative_diff = label_diff / label_base

    # 正样本：标签差值小，且属于同一 domain
    pos_mask = (relative_diff < label_thresh_ratio) & domain_mask
    neg_mask = (relative_diff >= label_thresh_ratio) & domain_mask
    diag_mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = pos_mask & diag_mask
    neg_mask = neg_mask & diag_mask

    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # 相似度矩阵：用于对比学习
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # 高斯核加权（仅对正样本）
    weight_matrix = torch.exp(- (label_diff ** 2) / (2 * sigma ** 2))
    pos_weight = weight_matrix.masked_fill(~pos_mask, 0.0)

    # 加权 InfoNCE 损失
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = (torch.exp(pos_sim / temperature) * pos_weight).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()




