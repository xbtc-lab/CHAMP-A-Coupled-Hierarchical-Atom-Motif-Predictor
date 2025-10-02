import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import rdkit.Chem as Chem
import warnings
import os

def reg_visual_umap(X, y,argse,epoch):
    # 确保输入是 NumPy 数组
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # 如果 X 是 torch.Tensor，转换为 NumPy 数组
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # 如果 y 是 torch.Tensor，转换为 NumPy 数组

    warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value.*")
    # 将高维特征降维到二维空间（使用 UMAP）
    umap = UMAP(n_components=2, random_state=42)
    X_2d = umap.fit_transform(X)

    # 归一化目标值 y 到 [-1, 1] 范围
    y_normalized = (y - y.min()) / (y.max() - y.min()) * 2 - 1

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_normalized, cmap='coolwarm', s=5, alpha=0.8)

    # 添加颜色条（Colorbar），解释颜色与数值的对应关系
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Target Value', rotation=270, labelpad=15)

    # 设置标题和坐标轴标签
    plt.title(f"Regression Task Visualization (UMAP)-{argse.dataset}:{epoch+1}")
    plt.xlabel("Dimension 1 (UMAP)")
    plt.ylabel("Dimension 2 (UMAP)")

    path = f"./Experiment/reg_Image/{argse.dataset}"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./Experiment/reg_Image/{argse.dataset}/{epoch}_{argse.use_head_interaction}_{argse.use_gating}_UMAP.png")
    # 显示图形
    plt.show()

def reg_visual_pca(X, y,argse,epoch):
    # 确保输入是 NumPy 数组
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # 如果 X 是 torch.Tensor，转换为 NumPy 数组
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # 如果 y 是 torch.Tensor，转换为 NumPy 数组

    warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value.*")
    # 将高维特征降维到二维空间（使用 UMAP）
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # 归一化目标值 y 到 [-1, 1] 范围
    y_normalized = (y - y.min()) / (y.max() - y.min()) * 2 - 1

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_normalized, cmap='coolwarm', s=5, alpha=0.8)

    # 添加颜色条（Colorbar），解释颜色与数值的对应关系
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Target Value', rotation=270, labelpad=15)

    # 设置标题和坐标轴标签
    plt.title(f"Regression Task Visualization (PCA)-{argse.dataset}:{epoch+1}")
    plt.xlabel("Dimension 1 (PCA)")
    plt.ylabel("Dimension 2 (PCA)")

    path = f"./Experiment/reg_Image/{argse.dataset}"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./Experiment/reg_Image/{argse.dataset}/{epoch}_{argse.use_head_interaction}_{argse.use_gating}_PCA.png")
    # 显示图形
    plt.show()


def reg_visual_TSNE(X, y,argse,epoch):
    # 确保输入是 NumPy 数组
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # 如果 X 是 torch.Tensor，转换为 NumPy 数组
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # 如果 y 是 torch.Tensor，转换为 NumPy 数组

    warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value.*")
    # 将高维特征降维到二维空间（使用 UMAP）
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    # 归一化目标值 y 到 [-1, 1] 范围
    y_normalized = (y - y.min()) / (y.max() - y.min()) * 2 - 1

    # 绘制散点图
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_normalized, cmap='coolwarm', s=5, alpha=0.8)

    # 添加颜色条（Colorbar），解释颜色与数值的对应关系
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Target Value', rotation=270, labelpad=15)

    # 设置标题和坐标轴标签
    plt.title(f"Regression Task Visualization (TSNE)-{argse.dataset}:{epoch+1}")
    plt.xlabel("Dimension 1 (TSNE)")
    plt.ylabel("Dimension 2 (TSNE)")

    path = f"./Experiment/reg_Image/{argse.dataset}"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./Experiment/reg_Image/{argse.dataset}/{epoch}_{argse.use_head_interaction}_{argse.use_gating}_TSNE.png")
    # 显示图形
    plt.show()


def task_visual(X, Y, epoch):
    # 确保输入是 NumPy 数组
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # 如果 X 是 torch.Tensor，转换为 NumPy 数组
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()  # 如果 y 是 torch.Tensor，转换为 NumPy 数组
        Y = Y.flatten()

    umap = UMAP(n_components=2, random_state=42)
    embedding = umap.fit_transform(X)  # 降维到 2D
    unique_labels = np.unique(Y)

    # 绘制散点图
    plt.figure(figsize=(10, 8))

    # 定义颜色映射
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # 使用 tab10 调色板

    for i, label in enumerate(unique_labels):
        # 筛选出当前标签的数据点
        mask = Y == label
        plt.scatter(
            embedding[mask, 0],  # x 坐标
            embedding[mask, 1],  # y 坐标
            label=f'Class {label}',  # 图例标签
            color=colors(i),  # 颜色
            s=10,  # 点的大小
            alpha=0.7  # 透明度
        )


    # 移除 x 和 y 轴的刻度和边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xticks([])  # 移除 x 轴刻度
    plt.yticks([])  # 移除 y 轴刻度


    # 添加图例、标题和坐标轴标签
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'UMAP Projection of Data{epoch+1}', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tight_layout()

    # 显示图像
    plt.show()


def find_substructure_indices(molecule_smiles, substructure_smiles_list):
    """
    参数:
        molecule_smiles (str): 分子的 SMILES 表达式。
        substructure_smiles_list (list): 官能团的 SMILES 表达式列表。
    返回:
        list: 匹配成功的官能团索引列表。
    """
    # 将分子的 SMILES 转换为 RDKit 分子对象
    molecule = Chem.MolFromSmiles(molecule_smiles)
    if molecule is None:
        raise ValueError("Invalid molecule SMILES")

    # 初始化结果列表
    matched_indices = []

    # 遍历官能团列表
    for idx, substructure_smiles in enumerate(substructure_smiles_list):
        # 将官能团的 SMILES 转换为 RDKit 分子对象
        substructure = Chem.MolFromSmiles(substructure_smiles)
        if substructure is None:
            raise ValueError(f"Invalid substructure SMILES at index {idx}: {substructure_smiles}")

        # 检查官能团是否是分子的子结构
        if molecule.HasSubstructMatch(substructure):
            matched_indices.append(idx)

    return matched_indices


def plot_embeddings(data, method='tsne'):
    # 过滤掉 label 为 None 的数据
    filtered_data = [(label, tensor) for label, tensor in data if label is not None]

    if not filtered_data:
        print("No valid data to plot.")
        return

    labels, tensors = zip(*filtered_data)
    labels = list(labels)
    tensor_matrix = torch.stack(tensors)  # shape: (N, D)

    # 降维为2维
    if method == 'pca':
        reduced = PCA(n_components=2).fit_transform(tensor_matrix.cpu().numpy())
    elif method == 'tsne':
        reduced = TSNE(n_components=2, random_state=42).fit_transform(tensor_matrix.cpu().numpy())
    elif method == 'umap':
        reduced = UMAP(n_components=2).fit_transform(tensor_matrix.cpu().numpy())
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    # 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.8)
    plt.colorbar(scatter, ticks=sorted(set(labels)))
    plt.title(f'2D Scatter Plot ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    with open("","rb") as file:
        pickle.load(file)
