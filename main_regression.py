import pickle
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv,global_add_pool,global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import numpy as np
import math

import Args
from Model import TDL_CCL
from Model import PABME
from motif_extract import mol_motif, motif_graph
from Model.DataProcessing import MoleculeMotifDataset
from Model.atom_motif_attention import AtomMotifAttention
from Model.HMSAF import HMSAF

from Model import utils
import importlib
importlib.reload(TDL_CCL)
importlib.reload(utils)
importlib.reload(PABME)
importlib.reload(mol_motif)
importlib.reload(motif_graph)


def global_atom_attr(data):
    motif_batch = data['motif'].batch    # 每个 motif 所属的图索引
    atom_ptr = data['atom'].ptr          # 每个图的 atom 索引起点

    # 获取边的连接信息和属性
    edge_index = data['motif', 'connects', 'motif'].edge_index
    edge_attr = data['motif', 'connects', 'motif'].edge_attr

    # 确定每条边所属的图
    src_motif = edge_index[0]  # 源 motif 索引
    graph_idx = motif_batch[src_motif]  # 每条边所属的图索引

    # 从 edge_attr 中提取本地 atom 索引
    local_atom_idx_src = edge_attr[:, -1]  # 源节点的本地索引
    local_atom_idx_dst = edge_attr[:, -2]  # 目标节点的本地索引

    # 计算全局 atom 索引
    offsets = atom_ptr[graph_idx]  # 每条边对应图的 atom 索引偏移
    global_atom_idx_src = local_atom_idx_src + offsets
    global_atom_idx_dst = local_atom_idx_dst + offsets

    # 构建新的 edge_attr：特征 + 全局 atom 索引 + 边的全局索引
    new_edge_attr = torch.stack([global_atom_idx_src, global_atom_idx_dst],dim=1)
    return new_edge_attr



# Message Passing of motifs
class GINENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GINENet, self).__init__()
        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        ), edge_dim=edge_dim)

        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        ), edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x


# Heterogeneous GNN for encoding motifs
class MotifGIN(torch.nn.Module):
    def __init__(self, node_dim,edge_dim,type_dim,hidden_dim,num_layers=2):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim

        # Initial embedding layers
        self.atom_encoder = nn.Sequential(
                            Linear(node_dim, hidden_dim),
                            ReLU())

        self.edge_encoder = nn.Sequential(
                            Linear(edge_dim, hidden_dim),
                            ReLU())

        self.motif_type_encoder = nn.Embedding(4, type_dim)  # Assuming 4 motif types

        # Internal motif processing using MotifGINLayer
        self.motif_GIN = PABME.GNNModel(hidden_dim, num_layers)

        self.motif_node_nn = nn.Linear(hidden_dim + type_dim,hidden_dim)

        self.motif_edge_nn = nn.Linear(hidden_dim * 2 + type_dim,hidden_dim)

        self.motif_Message_Passing = GINENet(in_channels=hidden_dim,
                                             hidden_channels=hidden_dim,
                                             out_channels=hidden_dim,
                                             edge_dim=hidden_dim)

    def forward(self, data):
        # # ==================Prepare embeddings# ==================
        # Encode atoms
        x_atom = self.atom_encoder(data['atom'].x.float())

        # Encode motifs
        motif_type = data['motif'].type
        motif_type_embedding = self.motif_type_encoder(motif_type)

        # Encode edges - atom-atom edges
        edge_index = data[("atom", "motif_internal", "atom")].edge_index
        edge_attr= self.edge_encoder(data[('atom', 'motif_internal', 'atom')].edge_attr.float())

        # ==================encoding node of motifs: type of motif and edge and atom====================
        # input:x, edge_index, edge_attr, motif_atom_edge_index
        motif_atom_edge_index = data["motif", "contains", "atom"].edge_index

        _,_,h_motif_atom, x = self.motif_GIN(x_atom, edge_index, edge_attr, motif_atom_edge_index)
        h_motif_atom = self.motif_node_nn(torch.cat((motif_type_embedding, h_motif_atom), dim=1))

        # ==================encoding edge of motifs: type of motif and edge and atom====================
        motif_edge_attr = torch.cat([data["motif", "connects", "motif"].edge_attr[:,:-2],global_atom_attr(data)],dim=1)    # 用于修改edge_attr中的node（local->全局）
        atom_edge_dim = data["atom", "motif_internal", "atom"].edge_attr.shape[1]

        # 1.type of motif
        src_motif_type = motif_edge_attr[:,0].long()
        dis_motif_type = motif_edge_attr[:,1].long()
        couple_motifs_type = self.motif_type_encoder(src_motif_type) + self.motif_type_encoder(dis_motif_type)

        # 2.edge and atom
        node_indices = motif_edge_attr[:,-2:].long()
        node_embeddings = torch.index_select(x_atom, 0, node_indices[:,0]) + torch.index_select(x_atom, 0, node_indices[:,1])
        edge_embeddings = self.edge_encoder(motif_edge_attr[:,2:2 + atom_edge_dim].float())

        # Combine embeddings
        h_motif_edge_attr = self.motif_edge_nn(torch.cat([couple_motifs_type, edge_embeddings, node_embeddings], dim=1))

        # ==================Message Passing of motifs:h_motif_atom and h_motif_edge_attr====================
        motif_edge_index = data["motif", "connects", "motif"].edge_index
        h_motif_atom = self.motif_Message_Passing(h_motif_atom,motif_edge_index, h_motif_edge_attr)

        # ============================Pool of motif=============================
        motif_level = global_add_pool(h_motif_atom,batch = data["motif"].batch)

        return h_motif_atom , x_atom , motif_level

class MotifBasedModel(torch.nn.Module):
    def __init__(self,node_feature_dim,edge_feature_dim,hidden_dim):
        super(MotifBasedModel, self).__init__()

        self.atom_encoder = nn.Sequential(Linear(node_feature_dim, hidden_dim),ReLU())

        # Heterogeneous GNN for joint atom-motif representation learning
        self.motif_gin = MotifGIN(node_feature_dim, edge_feature_dim, 16,hidden_dim, num_layers=2)

        # self.GIN_atom = GINE.ImprovedGINE(node_feature_dim=node_feature_dim, edge_feature_dim=edge_feature_dim, hidden_dim=hidden_dim)

        # atom_motif_attention
        self.atom_motif_attn = AtomMotifAttention(atom_dim=hidden_dim,
                                                  motif_dim=hidden_dim,
                                                  num_heads=4,
                                                  dropout = 0.2)

        self.DCM_attention = HMSAF(
                                  n_head = 4,
                                  input_dim = hidden_dim,
                                  output_dim = hidden_dim,
                                  use_head_interaction = argse.use_head_interaction,
                                  use_gating = argse.use_gating,
                                  dropout=0.2)

        # 添加层归一化以稳定训练
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)


        self.motif_read_out = nn.Linear(hidden_dim, 1)
        self.atom_read_out = nn.Linear(hidden_dim, 1)

        # 初始alpha和beta权重为0.5，可学习
        # 共享权重的多层感知机
        reduction_ratio = 4
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, (hidden_dim + hidden_dim) // reduction_ratio),
            nn.ReLU(),
            nn.Linear((hidden_dim + hidden_dim) // reduction_ratio, hidden_dim),
            nn.Sigmoid()  # 输出[0,1]的通道权重
        )

        # 粗细粒度的门阀控制
        # self.gate = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),  # 输入拼接后的特征
        #     nn.Sigmoid()  # 输出[0,1]的权重
        # )

        # # 添加额外的融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.1)
        )
        #
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.beta = nn.Parameter(torch.tensor(0.5))

        # MIB:
        # self.MIB = VIB.DualVIB(input_dim = hidden_dim,
        #                        hidden_dim = hidden_dim,
        #                        latent_dim = 16)
        # self.MINE = VIB.MINE(latent_dim = 16)

    def forward(self, data):
        # -------------------------------
        # 1. compute embedding of motif
        # -------------------------------
        h_motif_atom, x_atom, motif_level = self.motif_gin(data)


        # 更新图数据
        data["motif"].x = h_motif_atom
        # TODO 1:修改了这一处：如果使用计算motif_embedding的继续使用，可能会出问题
        # data["atom"].x = x_atom                                      # 改进前：
        data["atom"].x = self.atom_encoder(data["atom"].x.float())     # 改进后：
        # data["atom"].x = self.GIN_atom(data["atom"].x,data["atom","bond","atom"].edge_index,data["atom","bond","atom"].edge_attr)

        # -------------------------------
        # Stage 2: 粗粒度特征 - atom-motif注意力
        # -------------------------------
        hetero_data = self.atom_motif_HeteroData(data)
        coarse_particle_feature, motif_to_atom_attn = self.atom_motif_attn.get_atom_to_atom_attention_efficient(hetero_data)
        coarse_particle_feature = self.layer_norm1(coarse_particle_feature)           # 应用层归一化


        # -------------------------------
        # Stage 3: 细粒度特征 - 节点间动态注意力
        # -------------------------------
        fine_particle_feature = self.DCM_attention(data["atom"].x, data["atom"].batch, motif_to_atom_attn)
        fine_particle_feature = self.layer_norm2(fine_particle_feature)               # 应用层归一化

        # -------------------------------
        # Stage 4: 融合粗细粒度特征:
        # -------------------------------
        # 方法1: 门控机制
        # TODO 2:修改门阀控制
        combined = torch.cat([coarse_particle_feature, fine_particle_feature], dim=-1)
        # 生成通道注意力权重
        channel_weights = self.mlp(combined)
        # 加权融合
        atom = channel_weights * coarse_particle_feature + (1 - channel_weights) * fine_particle_feature


        # 修改前：
        # combined = torch.cat([coarse_particle_feature, fine_particle_feature], dim=-1)
        # gate_weight = self.gate(combined)
        # gated_feature = gate_weight * coarse_particle_feature + (1 - gate_weight) * fine_particle_feature

        # weighted_avg = self.alpha * coarse_particle_feature + self.beta * fine_particle_feature # 自适应加权平均
        # atom = gated_feature + data["atom"].x

        # 额外的特征融合
        atom = self.fusion(atom)

        # -------------------------------
        # 5.MIB:不同粒度的多模态信息瓶颈
        # -------------------------------
        # mu_a, logvar_a, mu_m, logvar_m, z_atom, z_motif = self.MIB(coarse_particle_feature,fine_particle_feature)
        # # MIB 互信息损失
        # MIB_loss = VIB.loss_function(mu_a, logvar_a, mu_m, logvar_m, z_atom, z_motif, self.MINE, gamma=1e-2)


        # 读出层
        atom_level = global_mean_pool(atom, data["atom"].batch)    # 表示每个分子图的高维特征表示
        y_atom = self.atom_read_out(atom_level)
        y_motif = self.motif_read_out(motif_level)


        # return y_atom, y_motif, MIB_loss, {'gate': channel_weights.mean().item(), 'alpha': self.alpha.item(), 'beta': self.beta.item()}
        return y_atom, y_motif, {'gate': channel_weights.mean().item(),"atom_level":atom_level}

    def atom_motif_HeteroData(self,data):
        hetero_data = HeteroData()

        # Add atom features
        hetero_data['atom'].x = data['atom'].x
        hetero_data['atom'].batch = data['atom'].batch

        # Add motif features
        hetero_data['motif'].x = data['motif'].x
        hetero_data['motif'].batch = data['motif'].batch

        # Add atom-in-motif edges
        hetero_data['atom', 'in', 'motif'].edge_index = data['atom', 'in', 'motif'].edge_index
        return hetero_data

# 修改训练函数，添加损失权重动态调整和梯度分析
def train(model, loader, optimizer, criterion, device, argse, check_grad=False, epoch=0):
    model.train()
    total_loss = 0
    atom_loss_total = 0
    motif_loss_total = 0
    gate_values = 0
    alpha_values = 0
    beta_values = 0
    processed_batches = 0

    # Create progress bar
    pbar = tqdm(loader, desc='Training')

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # Move batch to device.
        batch = batch.to(argse.device)

        # Forward pass
        out_atom, out_motif, metrics = model(batch)
        y = batch["mol"].y

        # 计算损失
        loss_atom = criterion(out_atom, y)  # 原子级别的损失
        loss_motif = criterion(out_motif, y)  # 子图级别的损失
        if argse.is_contrastive:
            loss_contrastive_ring = TDL_CCL.compute_ring_contrastive_loss_regression(batch,label_thresh_ratio=argse.label_thresh_ratio)
            loss_contrastive_noring = TDL_CCL.compute_nonring_contrastive_loss_regression(batch,label_thresh_ratio=argse.label_thresh_ratio)
            loss = loss_atom + loss_motif + argse.alpha * loss_contrastive_ring + argse.beta * loss_contrastive_noring
        else:
            loss = loss_atom + loss_motif
        # loss = loss_atom + loss_motif + 0.2 * loss_contrastive_ring + 0.2 * loss_contrastive_noring
        # loss = loss_atom + loss_motif
        # 添加L2正则化以防止过拟合
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)

        loss += 1e-5 * l2_reg  # 轻微的L2正则化

        # 检查损失值是否有效
        if not torch.isfinite(loss):
            print(f"警告: 无效损失值，跳过此批次")
            continue

        loss.backward()

        # 梯度裁剪以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录损失
        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        atom_loss_total += loss_atom.item() * batch_size
        motif_loss_total += loss_motif.item() * batch_size

        # 记录门控和权重值:
        gate_values += metrics['gate'] * batch_size
        # alpha_values += metrics['alpha'] * batch_size
        # beta_values += metrics['beta'] * batch_size

        processed_batches += 1

        # 更新进度条信息
        pbar.set_postfix({
            'batch_loss': f'{loss.item():.4f}',
            'atom_loss': f'{loss_atom.item():.4f}',
            'motif_loss': f'{loss_motif.item():.4f}',
            'gate': f'{metrics["gate"]:.2f}'
        })

    if processed_batches > 0:
        avg_samples = processed_batches * loader.batch_size
        return {
            'loss': total_loss / avg_samples,
            'atom_loss': atom_loss_total / avg_samples,
            'motif_loss': motif_loss_total / avg_samples,
            'gate': gate_values / processed_batches,
            # 'alpha': alpha_values / processed_batches,
            # 'beta': beta_values / processed_batches
        }
    else:
        return {'loss': float('inf'), 'atom_loss': float('inf'), 'motif_loss': float('inf')}


# 修改验证函数以匹配新的训练函数格式
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    atom_loss_total = 0
    motif_loss_total = 0
    total_rmse = 0
    processed_batches = 0
    total_samples = 0
    gate_values = 0
    alpha_values = 0
    beta_values = 0
    atom_level = torch.empty(0,32).to(argse.device)
    y_atom = torch.empty(0,1).to(argse.device)

    # Create progress bar for evaluation
    pbar = tqdm(loader, desc='Evaluating')

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(argse.device)

        out_atom, out_motif, metrics = model(batch)
        y = batch["mol"].y

        loss_atom = criterion(out_atom, y)
        loss_motif = criterion(out_motif, y)

        # 验证时等权重
        loss = loss_atom * 0.5 + loss_motif * 0.5

        # 记录总损失
        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        atom_loss_total += loss_atom.item() * batch_size
        motif_loss_total += loss_motif.item() * batch_size

        # 计算误差平方和（RMSE）
        squared_error_atom = torch.pow(out_atom - y, 2).sum().item()
        total_rmse += squared_error_atom

        # 记录门控和权重值
        gate_values += metrics['gate'] * batch_size
        # alpha_values += metrics['alpha'] * batch_size
        # beta_values += metrics['beta'] * batch_size

        # 记录每一个分子的高维特征表示
        atom_level = torch.cat((atom_level, metrics["atom_level"]), dim = 0)
        y_atom = torch.cat((y_atom, y), dim = 0)

        total_samples += batch_size
        processed_batches += 1

        # Update progress bar
        avg_loss = total_loss / total_samples
        avg_rmse = (total_rmse / total_samples) ** 0.5
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'rmse': f'{avg_rmse:.4f}'
        })

    if processed_batches > 0:
        return total_loss / total_samples, (total_rmse / total_samples) ** 0.5, {
            'atom_loss': atom_loss_total / total_samples,
            'motif_loss': motif_loss_total / total_samples,
            'gate': gate_values / processed_batches,
            "atom_level":atom_level,
            "y_atom":y_atom
            # 'alpha': alpha_values / processed_batches,
            # 'beta': beta_values / processed_batches
        }
    else:
        return float('inf'), float('inf'), {}


def dataloader(dataset):
    dataset = MoleculeMotifDataset(root="./dataset/", name=dataset)
    print(f"Dataset contains {len(dataset)} molecules")

    # Check if the processed dataset has the correct format
    sample_data = dataset[0]
    print("Checking first graph in dataset:")
    print(f"Motif types: {sample_data['motif'].type}")

    n = len(dataset)
    indices = list(range(n))

    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    train_index = indices[:train_size]
    val_index = indices[train_size:train_size + val_size]
    test_index = indices[train_size + val_size:]

    print(f"\nDataset split:")
    print(f"Train: {len(train_index)} samples")
    print(f"Validation: {len(val_index)} samples")
    print(f"Test: {len(test_index)} samples")

    # Create dataloaders
    def create_dataloader(indices, batch_size=32, shuffle=True):
        return DataLoader(
            dataset=[dataset[i] for i in indices],
            batch_size=batch_size,
            shuffle=shuffle
        )

    train_loader = create_dataloader(train_index, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_index, batch_size=32, shuffle=False)
    test_loader = create_dataloader(test_index, batch_size=32, shuffle=False)
    # 用于最后测试，画出散点图
    total_loader = create_dataloader(list(range(len(dataset))), batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader, total_loader


def main(train_loader, val_loader, test_loader,total_loader,argse):
    # Initialize model
    node_feature_dim = 9
    edge_feature_dim = 3
    hidden_dim = 32

    model = MotifBasedModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
    ).to(argse.device)


    # 调整优化器，使用较小的学习率
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.6, verbose=True
    )
    # 使用更稳定的损失函数
    criterion = nn.HuberLoss()  # Huber损失对异常值更有鲁棒性

    # Training loop
    best_val_loss = float('inf')
    print("\nStarting training...")
    print("-" * 50)

    import matplotlib.pyplot as plt

    # 初始化存储训练和验证结果的列表
    train_losses = []
    val_losses = []
    val_RMSEs = []
    test_RMSEs = []
    learning_rates = []

    # TODO：用于记录
    Poly3D = []

    for epoch in range(argse.epochs):
        print(f"\nEpoch {epoch + 1}/80")

        # 每5个epoch执行一次详细梯度检查
        check_grad = (epoch % 10 == 0)
        if check_grad:
            print("将对本轮进行详细梯度检查")

        # 传递epoch参数用于动态调整损失权重
        train_metrics = train(model, train_loader, optimizer, criterion, argse.device,argse, check_grad=check_grad, epoch=epoch)
        train_loss = train_metrics['loss']

        # 评估
        val_loss, val_RMSE, val_metrics = evaluate(model, val_loader, criterion, argse.device)
        _, test_RMSE, _ = evaluate(model, test_loader, criterion, argse.device)

        # 调整学习率 - 使用验证损失来调整学习率
        scheduler.step(val_loss)  # 添加这一行来更新学习率

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './best_model.pt')
            print(f"新的最佳模型已保存! (验证损失: {val_loss:.4f})")

        # 记录结果
        train_losses.append(train_loss if not math.isinf(train_loss) and not math.isnan(train_loss) else None)
        val_losses.append(val_loss)
        val_RMSEs.append(val_RMSE)
        test_RMSEs.append(test_RMSE)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 输出更详细的训练摘要
        print(f"Epoch {epoch:03d} 摘要:")
        print(f"  训练损失: {train_loss:.4f} (atom: {train_metrics['atom_loss']:.4f}, motif: {train_metrics['motif_loss']:.4f})")
        print(f"  验证损失: {val_loss:.4f} (atom: {val_metrics['atom_loss']:.4f}, motif: {val_metrics['motif_loss']:.4f})")
        print(f"  验证RMSE: {val_RMSE:.4f}")
        print(f"  测试RMSE: {test_RMSE:.4f}")
        print(f"  门控均值: {train_metrics['gate']:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        if (epoch + 1) % 10 == 0:
            Poly3D.append(test_RMSE)
            _, _, total_metrics = evaluate(model, total_loader, criterion, argse.device)

            data = {
                "atom_level":total_metrics["atom_level"],
                "y_atom":total_metrics["y_atom"],
                "dataset":argse.dataset,
                "epoch":epoch
            }

            output_dir = r"E:\postgraduate\learn_document\paper\化学\实验 (6)\Experiment\result_3\melt\scatter"
            with open(os.path.join(output_dir,f"{argse.dataset}_reg_visual_{epoch}.pkl"),"wb") as file:
                pickle.dump(data, file)

            utils.reg_visual_umap(total_metrics["atom_level"], total_metrics["y_atom"],argse.dataset,epoch)
            # utils.reg_visual_pca(total_metrics["atom_level"], total_metrics["y_atom"], argse.dataset, epoch)
            # utils.reg_visual_TSNE(total_metrics["atom_level"], total_metrics["y_atom"], argse.dataset, epoch)
    # ==========================================测试最终模型==========================================

    print("\nTesting best model...")

    model.load_state_dict(torch.load('./best_model.pt'))
    total_loss, total_RMSE, total_metrics = evaluate(model, total_loader, criterion, argse.device)
    test_loss, test_RMSE, test_metrics = evaluate(model, test_loader, criterion, argse.device)
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_RMSE:.4f}")

    # 绘制学习曲线
    plt.figure(figsize=(12, 8))

    # 子图 1: 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # 子图 2: 验证 RMSE 曲线
    plt.subplot(2, 1, 2)
    plt.plot(val_RMSEs, label="Validation RMSE", color="green")
    plt.plot(test_RMSEs, label="TEST RMSE", color="orange")
    plt.title("Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()

    # 显示图表
    plt.tight_layout()
    plt.show()
    #
    # # 可选：保存学习曲线为图片文件
    # plt.savefig("./learning_curves.png")
    return Poly3D,test_RMSE


if __name__ == "__main__":
    # Set the random seed

    import os
    import random


    def set_rng_seed(seed):
        random.seed(seed)  # 为 Python 设置随机种子
        np.random.seed(seed)  # 为 NumPy 设置随机种子
        torch.manual_seed(seed)  # 为 PyTorch 设置随机种子
        torch.cuda.manual_seed_all(seed)  # 如果使用 GPU，也设置 GPU 的随机种子
        torch.backends.cudnn.deterministic = True  # 禁用 cuDNN 的非确定性算法
        torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化
        # torch.use_deterministic_algorithms(True)  # 强制使用确定性算法

    # Set device

    # ==============================================用于画出lipo的回归热力图=============================================
    set_rng_seed(42)
    argse = Args.parse_args()
    argse.dataset = "Lipo"

    if argse.dataset == "Lipo":
        argse.batch_size = 32
        argse.label_thresh_ratio = 0.1

    train_loader, val_loader, test_loader, total_loader = dataloader(argse.dataset)
    Poly3D,test_auc = main(train_loader, val_loader, test_loader, total_loader, argse)





    # ==============================================用于计算ploy3D=============================================
    # results_dict = {}
    # for dataset in ["ESOL","Lipo","FreeSolv"]:
    #
    #     results = {}
    #     argse.dataset = dataset
    #
    #     if argse.dataset == "ESOL":
    #         argse.batch_size = 32
    #         argse.label_thresh_ratio = 0.1
    #
    #     elif argse.dataset == "Lipo":
    #         argse.batch_size = 32
    #         argse.label_thresh_ratio = 0.1
    #
    #     elif argse.dataset == "FreeSolv":
    #         argse.batch_size = 32
    #         argse.label_thresh_ratio = 0.3
    #
    #     for use_Guide in [True, False]:
    #         for use_gating in [True, False]:
    #             for use_head_interaction in [True, False]:
    #                 set_rng_seed(42)
    #                 argse.use_Guide = use_Guide
    #                 argse.use_gating = use_gating
    #                 argse.use_head_interaction = use_head_interaction
    #
    #                 train_loader, val_loader, test_loader, total_loader = dataloader(argse.dataset)
    #                 Poly3D,test_auc = main(train_loader, val_loader, test_loader, total_loader, argse)
    #
    #
    #                 # 存储：
    #                 results[f"{use_Guide}_{use_gating}_{use_head_interaction}"] = Poly3D
    #
    #     # 保存数据：
    #     output_dir = r"E:\postgraduate\learn_document\paper\化学\实验 (5)\Experiment\result_3\melt"
    #     with open(os.path.join(output_dir,f"{dataset}.pkl"),"wb") as file:
    #         pickle.dump(results,file)
    #
    #     print(results)

    # ==============================================用于做对比学习的权重参数=============================================
    # results_dict = {}
    # for dataset in ["ESOL","Lipo","FreeSolv"]:
    #
    #     results = {}
    #     argse.dataset = dataset
    #
    #     for alpha in [0,0.2,0.4,0.6,0.8,1]:
    #         data = []
    #         for beta in [0, 0.2, 0.4, 0.6, 0.8,1]:
    #             set_rng_seed(42)
    #             argse.beta = beta
    #             argse.alpha = alpha
    #             train_loader, val_loader, test_loader, total_loader = dataloader(argse.dataset)
    #             test_auc = main(train_loader, val_loader, test_loader, total_loader, argse)
    #             data.append(test_auc)
    #         results.append(data)
    #
    #     results_dict[dataset] = results

    # print(results_dict)








