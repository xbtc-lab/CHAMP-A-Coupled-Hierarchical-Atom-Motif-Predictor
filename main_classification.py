# main_classification.py
import pickle
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler
from torch_scatter import scatter_add
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import math
import random

import Args
from Model import TDL_CCL
from Model import PABME
from Model import DataProcessing
from motif_extract import mol_motif, motif_graph
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
    motif_batch = data['motif'].batch    # Graph index for each motif
    atom_ptr = data['atom'].ptr          # Starting atom index for each graph

    # Get edge connection information and attributes
    edge_index = data['motif', 'connects', 'motif'].edge_index
    edge_attr = data['motif', 'connects', 'motif'].edge_attr

    # Determine which graph each edge belongs to
    src_motif = edge_index[0]  # Source motif index
    graph_idx = motif_batch[src_motif]  # Graph index for each edge

    # Extract local atom indices from edge_attr
    local_atom_idx_src = edge_attr[:, -1]  # Local index of source node
    local_atom_idx_dst = edge_attr[:, -2]  # Local index of target node

    # Calculate global atom indices
    offsets = atom_ptr[graph_idx]  # Atom index offset for each edge's graph
    global_atom_idx_src = local_atom_idx_src + offsets
    global_atom_idx_dst = local_atom_idx_dst + offsets

    # Build new edge_attr: features + global atom indices + global edge indices
    new_edge_attr = torch.stack([global_atom_idx_src, global_atom_idx_dst], dim=1)
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
    def __init__(self, node_dim, edge_dim, type_dim, hidden_dim, Pair_MLP, gnn_type, num_layers=2):
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

        # Internal motif processing using MotifGINLayer :TODO:motif_1
        self.motif_GIN = PABME.GNNModel(hidden_dim, num_layers, Pair_MLP=Pair_MLP, gnn_type=gnn_type)

        self.motif_node_nn = nn.Linear(hidden_dim + type_dim, hidden_dim)

        self.motif_edge_nn = nn.Linear(hidden_dim * 2 + type_dim, hidden_dim)

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
        edge_attr = self.edge_encoder(data[('atom', 'motif_internal', 'atom')].edge_attr.float())

        # ==================encoding node of motifs: type of motif and edge and atom====================
        # input:x, edge_index, edge_attr, motif_atom_edge_index
        motif_atom_edge_index = data["motif", "contains", "atom"].edge_index

        node_alpha, pair_alpha, h_motif_atom, x = self.motif_GIN(x_atom, edge_index, edge_attr, motif_atom_edge_index)
        h_motif_atom = self.motif_node_nn(torch.cat((motif_type_embedding, h_motif_atom), dim=1))

        # ==================encoding edge of motifs: type of motif and edge and atom====================
        motif_edge_attr = torch.cat(
            [data["motif", "connects", "motif"].edge_attr[:, :-2], global_atom_attr(data)], dim=1)  # Modify edge_attr node (local->global)
        atom_edge_dim = data["atom", "motif_internal", "atom"].edge_attr.shape[1]

        # 1.type of motif
        src_motif_type = motif_edge_attr[:, 0].long()
        dis_motif_type = motif_edge_attr[:, 1].long()
        couple_motifs_type = self.motif_type_encoder(src_motif_type) + self.motif_type_encoder(dis_motif_type)

        # 2.edge and atom
        node_indices = motif_edge_attr[:, -2:].long()
        node_embeddings = torch.index_select(x_atom, 0, node_indices[:, 0]) + torch.index_select(x_atom, 0,
                                                                                                node_indices[:, 1])
        edge_embeddings = self.edge_encoder(motif_edge_attr[:, 2:2 + atom_edge_dim].float())

        # Combine embeddings
        h_motif_edge_attr = self.motif_edge_nn(torch.cat([couple_motifs_type, edge_embeddings, node_embeddings], dim=1))

        # ==================Message Passing of motifs:h_motif_atom and h_motif_edge_attr====================
        motif_edge_index = data["motif", "connects", "motif"].edge_index
        h_motif_atom = self.motif_Message_Passing(h_motif_atom, motif_edge_index, h_motif_edge_attr)

        # ============================Pool of motif=============================
        motif_level = global_add_pool(h_motif_atom, batch=data["motif"].batch)

        return node_alpha, pair_alpha, h_motif_atom, x_atom, motif_level


class MotifBasedModel(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, y_dim, Pair_MLP=True, gnn_type="our", ):
        super(MotifBasedModel, self).__init__()

        self.atom_encoder = nn.Sequential(Linear(node_feature_dim, hidden_dim), ReLU())

        # Heterogeneous GNN for joint atom-motif representation learning
        self.motif_gin = MotifGIN(node_feature_dim, edge_feature_dim, 16, hidden_dim, Pair_MLP=Pair_MLP,
                                  gnn_type=gnn_type, num_layers=2)

        # atom_motif_attention
        self.atom_motif_attn = AtomMotifAttention(atom_dim=hidden_dim, motif_dim=hidden_dim, num_heads=4, dropout=0.2)

        self.DCM_attention = HMSAF(n_head=4,
                                 input_dim=hidden_dim,
                                 output_dim=hidden_dim,
                                 use_Guide=True,
                                 use_head_interaction=True,
                                 use_gating=True)

        # Add layer normalization to stabilize training
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Modify output layer for binary classification
        self.motif_read_out = nn.Sequential(nn.Linear(hidden_dim, y_dim),
                                            # nn.Sigmoid()
                                            )
        self.atom_read_out = nn.Sequential(nn.Linear(hidden_dim, y_dim),
                                           # nn.Sigmoid()
                                           )

        # Initial alpha and beta weights at 0.5, learnable
        # Shared weight MLP
        reduction_ratio = 4
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, (hidden_dim + hidden_dim) // reduction_ratio),
            nn.ReLU(),
            nn.Linear((hidden_dim + hidden_dim) // reduction_ratio, hidden_dim),
            nn.Sigmoid()  # Output channel weights in [0,1]
        )

        # # Add additional fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, data):
        # -------------------------------
        # 1. compute embedding of motif
        # -------------------------------
        node_alpha, pair_alpha, h_motif_atom, x_atom, motif_level = self.motif_gin(data)

        # Save initial features for residual connection
        # atom_initial = x_atom

        # Update graph data
        data["motif"].x = h_motif_atom
        # TODO 1:Modified this part: If using computed motif_embedding, may cause issues
        # data["atom"].x = x_atom                                      # Before improvement:
        data["atom"].x = self.atom_encoder(data["atom"].x.float())  # After improvement:

        # -------------------------------
        # Stage 2: Coarse-grained features - atom-motif attention
        # -------------------------------
        hetero_data = self.atom_motif_HeteroData(data)
        coarse_particle_feature, motif_to_atom_attn = self.atom_motif_attn.get_atom_to_atom_attention_efficient(
            hetero_data)
        coarse_particle_feature = self.layer_norm1(coarse_particle_feature)  # Apply layer normalization

        # -------------------------------
        # Stage 3: Fine-grained features - inter-node dynamic attention
        # -------------------------------
        fine_particle_feature, attn_probs = self.DCM_attention(data["atom"].x, data["atom"].batch, motif_to_atom_attn)
        fine_particle_feature = self.layer_norm2(fine_particle_feature)  # Apply layer normalization

        # -------------------------------
        # Stage 4: Fuse coarse and fine-grained features:
        # -------------------------------
        # Method 1: Gating mechanism
        # TODO 2: Modify gate control
        combined = torch.cat([coarse_particle_feature, fine_particle_feature], dim=-1)
        # Generate channel attention weights
        channel_weights = self.mlp(combined)
        # Weighted fusion
        atom = channel_weights * coarse_particle_feature + (1 - channel_weights) * fine_particle_feature

        # Additional feature fusion
        atom = self.fusion(atom)

        # Readout layer
        atom_level = global_mean_pool(atom, data["atom"].batch)  # High-dimensional feature representation of each molecular graph

        y_atom = self.atom_read_out(atom_level)
        y_motif = self.motif_read_out(motif_level)

        return y_atom, y_motif, h_motif_atom, {'gate': channel_weights.mean().item(), "atom_level": atom_level,
                                               "node_alpha": node_alpha, "pair_alpha": pair_alpha, "attn_probs": attn_probs}

    def atom_motif_HeteroData(self, data):
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


def compute_motif_contrastive_loss(batch, temperature=0.1, eps=1e-8):
    """
    Only perform structure-aware contrastive learning loss on ring5 and ring6 motifs.
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch]  # [N]
    type_list = batch["motif"].type  # [N]

    # Number of atoms per motif
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)  # [N]

    # Construct motif_type
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]
    type_class = [type_prefix[int(t)] for t in type_list]

    # Keep only ring5 and ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = [m in allowed_types for m in motif_type]
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=z.device)

    # If not enough samples, return 0 directly
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # Filter valid samples
    z = z[valid_mask]
    labels = labels[valid_mask]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_mask[i]]

    # Convert to tensors
    labels = labels.view(-1, 1)
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)
    type_class_tensor = torch.tensor([hash(t) for t in type_class], device=z.device).view(-1, 1)

    # Similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # Masks
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    # Here we use label:
    label_eq = (labels == labels.T)
    label_neq = (labels != labels.T)
    type_eq = (motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1))
    class_eq = (type_class_tensor == type_class_tensor.T)

    pos_mask = label_eq & type_eq & diag_mask
    neg_mask = label_neq & class_eq & diag_mask

    # Contrastive loss
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    # Positive sample pairs
    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    # Negative sample pairs
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()


# Multi-label multi-classification
def masked_roc_auc(y_true, y_score):
    y_true_np = y_true.cpu().numpy()
    y_score_np = y_score.cpu().numpy()
    aucs = []
    for i in range(y_true_np.shape[1]):
        y_col = y_true_np[:, i]
        p_col = y_score_np[:, i]
        mask = ~np.isnan(y_col)
        if np.sum(mask) >= 2 and len(np.unique(y_col[mask])) > 1:  # Need at least two classes to be meaningful
            auc = roc_auc_score(y_col[mask], p_col[mask])
            aucs.append(auc)
    return np.mean(aucs) if aucs else float('nan')


# Modify training function, add loss weight dynamic adjustment and gradient analysis
def train(model, loader, optimizer, criterion, device, argse, check_grad=False, epoch=0):
    model.train()
    total_loss = 0
    atom_loss_total = 0
    motif_loss_total = 0
    gate_values = 0
    alpha_values = 0
    beta_values = 0
    processed_batches = 0

    # For calculating classification accuracy
    total_correct = 0
    total_samples = 0

    # Create progress bar
    pbar = tqdm(loader, desc='Training')

    for batch_idx, batch in enumerate(pbar):
        try:
            optimizer.zero_grad()

            # Move batch to device.
            batch = batch.to(device)

            # Forward pass
            out_atom, out_motif, h_motif_atom, metrics = model(batch)
            y = batch["mol"].y

            # Calculate loss
            mask = ~torch.isnan(y)
            loss_atom = criterion(out_atom[mask], y[mask])  # Atom-level loss
            loss_motif = criterion(out_motif[mask], y[mask])  # Subgraph-level loss

            if argse.is_contrastive == True:
                contrastive_loss_ring = TDL_CCL.compute_ring_contrastive_loss_multilabel(batch,
                                                                                                     temperature=0.1,
                                                                                                     eps=1e-8)
                contrastive_loss_noring = TDL_CCL.compute_nonring_contrastive_loss_multilabel(batch,
                                                                                                          temperature=0.1,
                                                                                                          threshold=argse.label_thresh_ratio,
                                                                                                          eps=1e-8)
                loss = loss_atom + loss_motif + argse.alpha * contrastive_loss_ring + argse.beta * contrastive_loss_noring
            # Calculate contrastive loss
            else:
                loss = loss_atom + loss_motif

            # Add L2 regularization to prevent overfitting
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)

            loss += 1e-5 * l2_reg  # Light L2 regularization

            # Check if loss value is valid
            if not torch.isfinite(loss):
                print(f"Warning: Invalid loss value, skipping this batch")
                continue

            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Calculate classification accuracy
            # predictions = torch.sigmoid(out_atom) > 0.5
            # total_correct += (predictions == y).sum().item()
            # total_samples += y.size(0)

            preds = torch.sigmoid(out_atom) >= 0.5
            mask_np = ~torch.isnan(y)
            total_correct += (preds == y)[mask_np].sum().item() / mask.sum().item()
            total_samples += y.size(0)

            # Record losses
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            atom_loss_total += loss_atom.item() * batch_size
            motif_loss_total += loss_motif.item() * batch_size

            # Record gate and weight values:
            gate_values += metrics['gate'] * batch_size
            # alpha_values += metrics['alpha'] * batch_size
            # beta_values += metrics['beta'] * batch_size

            processed_batches += 1

            # Update progress bar information
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'atom_loss': f'{loss_atom.item():.4f}',
                'motif_loss': f'{loss_motif.item():.4f}',
                'gate': f'{metrics["gate"]:.4f}',
                'acc': f'{total_correct / total_samples:.4f}'
            })

        except Exception as e:
            print(e)

    if processed_batches > 0:
        avg_samples = total_samples
        return {
            'loss': total_loss / avg_samples,
            'atom_loss': atom_loss_total / avg_samples,
            'motif_loss': motif_loss_total / avg_samples,
            'gate': gate_values / processed_batches,
            'accuracy': total_correct / processed_batches * 100  # Display as percentage
            # 'alpha': alpha_values / processed_batches,
            # 'beta': beta_values / processed_batches
        }
    else:
        return {'loss': float('inf'), 'atom_loss': float('inf'), 'motif_loss': float('inf'), 'accuracy': 0.0}


# Modify validation function to match new training function format
@torch.no_grad()
def evaluate(model, loader, criterion, device, end=False):
    model.eval()
    total_loss = 0
    atom_loss_total = 0
    motif_loss_total = 0
    total_samples = 0
    gate_values = 0
    processed_batches = 0

    # For ROC-AUC calculation of true labels and predicted probabilities
    all_labels = []
    all_probs = []
    roc_auc_list = []
    contrastive_data = []

    # For other metrics
    atom_level = torch.empty(0, 32).to(device)
    y_atom = torch.empty(0, 1).to(device)

    # Count correctly predicted samples
    correct_num = 0
    sample_num = 0
    # Create progress bar for evaluation
    pbar = tqdm(loader, desc='Evaluating')

    for batch_idx, batch in enumerate(pbar):
        try:
            batch = batch.to(device)

            out_atom, out_motif, _, metrics = model(batch)
            y = batch["mol"].y

            mask = ~torch.isnan(y)
            loss_atom = criterion(out_atom[mask], y[mask])  # Atom-level loss
            loss_motif = criterion(out_motif[mask], y[mask])  # Subgraph-level loss
            loss = loss_atom * 0.5 + loss_motif * 0.5

            # Record total loss
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            atom_loss_total += loss_atom.item() * batch_size
            motif_loss_total += loss_motif.item() * batch_size

            preds = torch.sigmoid(out_atom) >= 0.5
            mask_np = ~torch.isnan(y)
            correct_num += (preds == y)[mask_np].sum().item()
            sample_num += mask_np.sum().item()

            roc_auc = masked_roc_auc(y, out_atom)
            if not np.isnan(roc_auc):
                roc_auc_list.append(roc_auc)

            # Record gate and weight values
            gate_values += metrics['gate'] * batch_size

            # Record high-dimensional feature representation of each molecule
            atom_level = torch.cat((atom_level, metrics["atom_level"]), dim=0)
            y_atom = torch.cat((y_atom, y), dim=0)

            total_samples += batch_size
            processed_batches += 1

            # Update progress bar
            avg_loss = total_loss / total_samples
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}'
            })

        except Exception as e:
            print(e)

    # Calculate overall accuracy
    accuracy = correct_num / sample_num * 100  # Display as percentage
    roc_auc = np.mean(roc_auc_list)

    if processed_batches > 0:
        return total_loss / total_samples, roc_auc, {
            'atom_loss': atom_loss_total / total_samples,
            'motif_loss': motif_loss_total / total_samples,
            'gate': gate_values / processed_batches,
            'accuracy': accuracy,
            "atom_level": atom_level,
            "y_atom": y_atom,
            "contrastive_data": contrastive_data
        }
    else:
        return float('inf'), 0.0, {}


def compute_pos_weight(dataset):
    labels = []
    for data in dataset:
        labels.append(data["mol"].y)

    labels = torch.cat(labels, dim=0)
    mask = ~torch.isnan(labels)
    pos = (labels[mask] == 1).sum().float()
    neg = (labels[mask] == 0).sum().float()
    pos_weight = neg / (pos + 1e-8)
    return pos_weight


def Data_loader(argse):
    # Load the heterogeneous graph dataset
    dataset = DataProcessing.MoleculeMotifDataset(root="./dataset/", name=argse.dataset)
    print(f"Dataset contains {len(dataset)} molecules")
    # Check if the processed dataset has the correct format
    sample_data = dataset[0]
    print("Checking first graph in dataset:")
    print(f"Motif types: {sample_data['motif'].type}")

    n = len(dataset)
    indices = list(range(n))
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    random.shuffle(indices)

    train_index = indices[:train_size]
    val_index = indices[train_size:train_size + val_size]
    test_index = indices[train_size + val_size:]
    total_index = indices[:]

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

    pos_weight = compute_pos_weight(dataset).to(argse.device)

    train_loader = create_dataloader(train_index, batch_size=argse.batch_size, shuffle=True)
    val_loader = create_dataloader(val_index, batch_size=argse.batch_size, shuffle=False)
    test_loader = create_dataloader(test_index, batch_size=argse.batch_size, shuffle=False)
    total_loader = create_dataloader(total_index, batch_size=argse.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, total_loader, sample_data, pos_weight


def main(train_loader, val_loader, test_loader, total_loader, sample_data, pos_weight, argse, gnn_type="our"):
    node_feature_dim = 9
    edge_feature_dim = 3
    hidden_dim = 32
    HIV_XY = list()
    y_dim = sample_data["mol"].y.shape[1]
    Pair_MLP = argse.Pair_MLP

    model = MotifBasedModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        y_dim=y_dim,
        # argse=argse,
        Pair_MLP=Pair_MLP,
        gnn_type=gnn_type,
    ).to(argse.device)

    # Adjust optimizer, use smaller learning rate
    optimizer = Adam(model.parameters(), lr=argse.lr, weight_decay=argse.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=argse.patience, factor=argse.factor, verbose=True
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_val_loss = float('inf')  # Change back to using validation loss to save best model
    best_val_auc = 0.0  # Still record best AUC
    print("\nStarting training...")
    print("-" * 50)

    import matplotlib.pyplot as plt

    # Initialize lists to store training and validation results
    train_losses = []
    val_losses = []
    val_aucs = []  # Modified to AUC
    train_accs = []
    val_accs = []
    learning_rates = []
    test_aucs = []
    scatter_result = []
    for epoch in range(argse.epochs):
        print(f"\nEpoch {epoch + 1}/{argse.epochs}")

        # Perform detailed gradient check every 5 epochs
        check_grad = (epoch % 10 == 0)
        if check_grad:
            print("Will perform detailed gradient check for this round")

        # Pass epoch parameter for dynamic loss weight adjustment
        train_metrics = train(model, train_loader, optimizer, criterion, argse.device, argse, check_grad=check_grad,
                              epoch=epoch)
        train_loss = train_metrics['loss']
        train_acc = train_metrics['accuracy']

        # Evaluate
        val_loss, val_roc_auc, val_metrics = evaluate(model, val_loader, criterion, argse.device)
        _, test_roc_auc, _ = evaluate(model, test_loader, criterion, argse.device)
        val_acc = val_metrics['accuracy']

        # Adjust learning rate
        scheduler.step(val_loss)

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc_at_best_loss = val_roc_auc
            torch.save(model.state_dict(),
                       f'E:/postgraduate/learn_document/paper/化学/实验 (6)/Experiment/result_2/node_pair/{argse.dataset}_{argse.Pair_MLP}.pt')
            print(
                f"New best model saved! (Validation loss: {val_loss:.4f}, Validation AUC: {val_roc_auc:.4f}%)")

        # Record best AUC (for tracking only, don't save model)
        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc

        # Record results
        train_losses.append(train_loss if not math.isinf(train_loss) and not math.isnan(train_loss) else None)
        val_losses.append(val_loss)
        val_aucs.append(val_roc_auc)
        test_aucs.append(test_roc_auc)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Output more detailed training summary
        print(f"Epoch {epoch:03d} Summary:")
        print(
            f"  Training loss: {train_loss:.4f} (atom: {train_metrics['atom_loss']:.4f}, motif: {train_metrics['motif_loss']:.4f})")
        print(f"  Training accuracy: {train_acc:.4f}%")
        print(
            f"  Validation loss: {val_loss:.4f} (atom: {val_metrics['atom_loss']:.4f}, motif: {val_metrics['motif_loss']:.4f})")
        print(f"  Validation ROC_AUC: {val_roc_auc:.4f}%")
        print(f"  Validation accuracy: {val_acc:.4f}%")
        print(f"  Test ROC_AUC: {test_roc_auc:.4f}%")
        print(f"  Gate mean: {train_metrics['gate']:.4f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    if argse.is_contrastive == False:
        output_file = f'umap_data_{argse.dataset}.pkl'
    else:
        output_file = f'umap_data_{argse.dataset}_contrast.pkl'

    with open(output_file, 'wb') as f:
        pickle.dump(scatter_result, f)

    print("\nTesting best model...")
    test_loss, test_auc, test_metrics = evaluate(model, test_loader, criterion, argse.device)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test ROC-AUC: {test_auc:.5f}%")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}%")

    # Plot learning curves
    plt.figure(figsize=(12, 12))

    # Subplot 1: Loss curves
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Subplot 2: Validation AUC curves
    plt.subplot(3, 1, 2)
    plt.plot(val_aucs, label="Validation ROC-AUC", color="red")
    plt.plot(test_aucs, label="Test ROC-AUC", color="green")
    plt.title("Validation ROC-AUC (%)")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC (%)")
    plt.legend()
    plt.grid()

    # Subplot 3: Accuracy curves
    plt.subplot(3, 1, 3)
    plt.plot(train_accs, label="Train Accuracy", color="blue")
    plt.plot(val_accs, label="Validation Accuracy", color="orange")
    plt.title(f"Training and Validation Accuracy (%)//{argse.alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    # Display charts
    plt.tight_layout()
    plt.show()

    # Save learning curves as image files
    plt.savefig("./learning_curves.png")
    return test_auc


if __name__ == "__main__":

    def set_rng_seed(seed):
        random.seed(seed)  # Set random seed for Python
        np.random.seed(seed)  # Set random seed for NumPy
        torch.manual_seed(seed)  # Set random seed for PyTorch
        torch.cuda.manual_seed_all(seed)  # If using GPU, also set GPU random seed
        torch.backends.cudnn.deterministic = True  # Disable
        torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-optimization


    set_rng_seed(47)
    argse = Args.parse_args()
    argse.dataset = "BBBP"

    train_loader, val_loader, test_loader, total_loader, sample_data, pos_weight = Data_loader(argse)
    test_auc = main(train_loader, val_loader, test_loader, total_loader, sample_data, pos_weight, argse)
