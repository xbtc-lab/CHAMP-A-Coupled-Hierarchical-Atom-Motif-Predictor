from collections import deque
from IPython.display import Image
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG       # 用于生成矢量图
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rdkit.Chem import Draw, rdmolops
import numpy as np
from rdkit import Chem
# from torch.distributed.rpc.api import method
from torch_geometric.datasets import MoleculeNet


# 可视化官能团
def visualize_motif(mol, fgs, method = "display"):
    # mol = Chem.MolFromSmiles(smiles)
    # 为每个原子设置标签
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetProp('atomNote', str(idx))  # 设置原子序号作为标签

    # 创建绘图对象
    # 用于保存：
    if method == "save":
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
        drawer.SetFontSize(6)  # 设置原子序号字体大小

    if method == "display":
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 300)
        drawer.SetFontSize(6)  # 设置原子序号字体大小


    # 为每个环分配颜色
    colors = list(mcolors.TABLEAU_COLORS.values())  # 使用matplotlib的颜色表
    # 字典
    highlight_atoms = {}

    for i, fg in enumerate(fgs):
        color = mcolors.to_rgb(colors[i % len(colors)])
        # 将set转换为list以便索引
        fg_list = list(fg)
        # 高亮环的原子
        for atom in fg_list:
            highlight_atoms[atom] = color

    # 绘制分子并高亮原子和键
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms.keys()),
        highlightAtomColors=highlight_atoms,
    )

    # 完成绘制
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

# 可是使用try-exception进行异常处理
# 区分芳香键和非芳香键
def visualize_ring_aromaticity(mol):
    # 2. 检测所有环并分类
    rings = mol.GetRingInfo().AtomRings()
    aromatic_rings = []
    non_aromatic_rings = []

    # 3. 判断每个环是否为芳香环
    for ring in rings:
        is_aromatic = True
        # 检查环内所有键是否都是芳香键
        bond_ids = []
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond and not bond.GetIsAromatic():
                is_aromatic = False
                break
        if is_aromatic:
            aromatic_rings.append(ring)
        else:
            non_aromatic_rings.append(ring)

    # 4. 准备可视化参数
    atom_colors = {}
    bond_colors = {}

    # 设置芳香环颜色（红色）
    for ring in aromatic_rings:
        for atom in ring:
            atom_colors[atom] = (1, 0, 0)  # RGB红色
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond:
                bond_colors[bond.GetIdx()] = (1, 0, 0)

    # 设置非芳香环颜色（绿色）
    for ring in non_aromatic_rings:
        for atom in ring:
            atom_colors[atom] = (0, 1, 0)  # RGB绿色
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond:
                bond_colors[bond.GetIdx()] = (0, 1, 0)
    # 加标签
    mol = Chem.Mol(mol)
    # 为每个原子设置标签
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetProp('atomNote', str(idx))  # 设置原子序号作为标签

    # 5. 生成SVG图像
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
    if len(rings)!=0:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=atom_colors.keys(),
            highlightAtomColors=atom_colors,
            highlightBonds=bond_colors.keys(),
            highlightBondColors=bond_colors
        )
        drawer.FinishDrawing()

    # 6. 显示结果
    svg = drawer.GetDrawingText().replace('svg:', '')
    return aromatic_rings, non_aromatic_rings, svg
    # return rings, non_aromatic_rings, svg

### 对满足非芳香键进行合并
def merge_aromatic_rings(rings):
    """
    合并满足条件的芳香键列表。

    参数:
        rings (list of tuple): 非芳香键列表，每个元组表示一个芳香键。

    返回:
        list of tuple: 合并后的芳香键列表。
    """
    # 将元组转换为集合以便操作
    rings = [set(ring) for ring in rings]

    # 标记是否需要进一步合并
    merged = True

    # 循环合并直到没有可以合并的环
    while merged:
        merged = False
        new_rings = []

        # 遍历每对环
        while rings:
            current_ring = rings.pop(0)
            merged_with_existing = False

            # 检查当前环是否可以与新环中的某个环合并
            for i, new_ring in enumerate(new_rings):
                # 如果两个环长度 >= 5 且共享元素 >= 2，则合并
                if len(current_ring) >= 5 and len(new_ring) >= 5 and len(current_ring & new_ring) >= 2:
                    new_rings[i] = current_ring | new_ring  # 合并环
                    merged_with_existing = True
                    merged = True
                    break

            # 如果没有合并，则将当前环添加到新环列表中
            if not merged_with_existing:
                new_rings.append(current_ring)

        # 更新环列表
        rings = new_rings

    # 将集合转换回元组
    return [tuple(sorted(ring)) for ring in rings]

def merge_single_h_neighbors(mol, merged_rings):
    """
    合并仅连接氢原子的单原子节点到相邻非芳香环中
    Args:
        mol: RDKit分子对象
        merged_rings: 合并后的非芳香环列表，每个元素是原子索引的集合
    Returns:
        更新后的非芳香环列表
    """
    new_rings = []

    for ring in merged_rings:
        extended_ring = set(ring)
        # 记录需要合并的原子
        candidates = set()

        # 遍历环内所有原子
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)

            # 遍历当前原子的所有邻居
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()

                # 跳过已经是环内的原子
                if neighbor_idx in ring:
                    continue

                # 检查该邻居是否满足条件
                # 条件1: 该原子所有非环连接的相邻原子都是氢
                # 条件2: 该原子本身不是氢
                if neighbor.GetAtomicNum() == 1:
                    continue  # 跳过氢原子本身

                all_hydrogen = True
                for nbr in neighbor.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    # 排除环连接点和自己
                    if nbr_idx == atom_idx or nbr_idx in ring:
                        continue
                    # 如果有非氢连接则不符合条件
                    if nbr.GetAtomicNum() != 1:
                        all_hydrogen = False
                        break

                if all_hydrogen:
                    candidates.add(neighbor_idx)

        # 合并符合条件的原子到当前环
        extended_ring.update(candidates)
        new_rings.append(frozenset(extended_ring))

    # 去重处理
    return [set(ring) for ring in list({ring for ring in new_rings})]

# 标记功能原子
def mark_functional_groups(mol, rings):
    """
    标记分子中的功能原子（杂原子、多重键碳、缩醛碳等），排除环结构中的原子。
    返回一个字典，键为原子索引，值为功能基团类型。
    """
    PATT = {
    'HETEROATOM': '[!#6]',                  # 匹配非碳原子（杂原子）
    'DOUBLE_TRIPLE_BOND': '*=,#*',        # 匹配双键（=）或三键（#）。
    # 'ACETAL': '[CX4]'                       # 初始SMARTS模式，用于初步筛选sp3碳
    }

    # 将SMARTS字符串转换为RDKit的mol对象
    PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}

    marks = []
    # 匹配是按照PATT的顺序匹配的
    for patt in PATT.values():
        for subs in mol.GetSubstructMatches(patt):
            subs = [sub for sub in subs if sub not in marks]
            for sub in subs :
                if sub not in rings:
                    marks.append(sub)

    # 匹配杂原子:
    # heteroatom_matches = mol.GetSubstructMatches(PATT['HETEROATOM'])
    # for match in heteroatom_matches:
    #     for atom_idx in match:
    #         if atom_idx not in rings:  # 排除环结构中的原子
    #             functional_atoms[atom_idx] = "Heteroatom"
    #
    # # 匹配多重键碳
    # double_triple_matches = mol.GetSubstructMatches(PATT['DOUBLE_TRIPLE_BOND'])
    # for match in double_triple_matches:
    #     for atom_idx in match:
    #         if atom_idx not in rings:  # 排除环结构中的原子
    #             print("C:", atom_idx)
    #             functional_atoms[atom_idx] = "Multiple Bond Carbon"
    #
    for atom in mol.GetAtoms():
        if atom.GetIdx() in rings:  # 排除环结构中的原子
            continue
        elif atom.GetTotalNumHs() == 0 and len([n for n in atom.GetNeighbors() if n.GetAtomicNum() not in [6, 1]]) >= 2:
            if atom.GetIdx() not in marks:
                marks.append(atom.GetIdx())
    #
    # # 现在的思路：利用一个一个的取，最后排序：
    #
    # # 设置优先级
    # """
    # (1)杂原子优先级最高
    # (2)多重键其次
    # (3)缩醛碳再次之
    # """
    # sorted_functional_atoms = dict(sorted(
    #     functional_atoms.items(),
    #     key=lambda x: (
    #         0 if x[1] == "Heteroatom" else  # 杂原子优先级最高
    #         1 if x[1] == "Multiple Bond Carbon" else  # 多重键次之
    #         2 if x[1] == "Acetal Carbon" else  # 缩醛碳再次之
    #         3  # 其他情况
    #     )
    # ))
    # print("sorted_functional_atoms:",sorted_functional_atoms)
    return marks

# 功能原子之间以及功能原子与非功能原子之间的合并
def merge_functional_groups(mol, marks, rings):
    """
    合并功能基团及其邻接碳原子为一个 motif，排除环结构中的原子。
    返回两个值：
    - fgs: 一个列表，每个元素是一个 motif 的原子索引集合。
    - adjacency_matrices: 一个列表，每个元素是一个 motif 的邻接矩阵。
    """
    fgs = []  # Function Groups
    adjacency_matrices = []  # 邻接矩阵列表

    node_visited = set()  # 针对功能 + 普通
    edge_visited = list()  # 针对功能 + 功能

    # 初始化：每个标记原子作为一个官能团
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx

    for atom in marks:  # init: each marked atom is a FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs) - 1]
        node_visited.add(atom)

    # 功能+功能 、 功能 + 非功能
    # 按照优先级处理功能原子及其邻接原子
    for atom_idx in marks:
        # 获取给定原子序号的原子的邻居原子
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()

            # 跳过环结构中的原子和已分配的原子
            if neighbor_idx in rings:
                continue

            # 如果邻接原子是功能原子，则合并它们所属的官能团（功能+功能）
            if neighbor_idx in marks:
                if {atom_idx, neighbor_idx} not in edge_visited:
                    assert len(atom2fg[atom_idx]) == 1 and len(atom2fg[neighbor_idx]) == 1
                    # 合并 neighbor_idx 的 FG 到 atom_idx 的 FG
                    fgs[atom2fg[atom_idx][0]].update(fgs[atom2fg[neighbor_idx][0]])     # 将邻居所在的官能团包含起来
                    fgs[atom2fg[neighbor_idx][0]] = set()
                    atom2fg[neighbor_idx] = atom2fg[atom_idx]
                    edge_visited.append({atom_idx, neighbor_idx})

            # 如果邻接原子是非功能原子，则将其加入当前 motif(功能 + 非功能)
            else:
                if neighbor_idx not in node_visited:
                    fgs[atom2fg[atom_idx][0]].add(neighbor_idx)
                    atom2fg[neighbor_idx].extend(atom2fg[atom_idx])
                    node_visited.add(neighbor_idx)

    # 清理空的官能团
    tmp = []
    for fg in fgs:
        if len(fg) == 0:
            continue
        tmp.append(fg)
    fgs = tmp

    # 构建每个官能团的邻接矩阵
    for fg in fgs:
        fg_list = sorted(fg)  # 确保原子索引有序
        size = len(fg_list)
        adj_matrix = np.zeros((size, size), dtype=int)  # 初始化邻接矩阵

        # 填充邻接矩阵
        for i, atom_idx in enumerate(fg_list):
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in fg:
                    j = fg_list.index(neighbor_idx)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # 对称填充

        adjacency_matrices.append(adj_matrix)

    return fgs, adjacency_matrices

# 判断给定一个 功能团和邻接矩阵，删除给定元素，判断是否连通
def is_connected_after_removal(fg, adjacency_matrix, removed_atom):
    """
    - True: 删除该原子后，剩余节点仍然连通。
    - False: 删除该原子后，剩余节点不再连通。
    """
    # 将官能团的原子序号转换为有序列表，并找到被删除原子的索引
    fg_list = sorted(fg)
    removed_index = fg_list.index(removed_atom)

    # 构建新的邻接矩阵，排除被删除的原子
    new_adjacency_matrix = np.delete(adjacency_matrix, removed_index, axis=0)  # 删除行
    new_adjacency_matrix = np.delete(new_adjacency_matrix, removed_index, axis=1)  # 删除列

    # 剩余节点的数量
    remaining_nodes = len(fg_list) - 1

    # 如果没有剩余节点，则直接返回 False
    if remaining_nodes == 0:
        return False

    # 使用 BFS 检查连通性
    visited = [False] * remaining_nodes  # 标记是否访问过
    queue = [0]  # 从第一个节点开始 BFS
    visited[0] = True

    while queue:
        current_node = queue.pop(0)
        for neighbor in range(remaining_nodes):
            if new_adjacency_matrix[current_node][neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # 如果所有剩余节点都被访问过，则连通；否则不连通
    return all(visited)

# 查找非环碳链
def find_non_ring_single_bond_only_carbon_chains_with_adjacency(mol):
    """找到分子中非环的单键碳链，且每个碳原子的所有键都必须是单键。

    参数:
        mol (rdkit.Chem.rdchem.Mol): RDKit分子对象

    返回:
        tuple: 包含两个列表 `(chains, chains_adjacency)`，
               其中：
                   - `chains` 是一个列表，存储所有符合条件的碳链；
                   - `chains_adjacency` 是一个列表，存储每个碳链对应的邻接矩阵。
    """
    # 1. 获取所有环原子
    rings = rdmolops.GetSSSR(mol)
    ring_atoms = set()
    for ring in rings:
        ring_atoms.update(ring)

    # 2. 筛选非环碳原子，且每个碳原子的所有键都必须是单键
    carbon_atoms = []
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 6 and
                atom.GetIdx() not in ring_atoms and
                all(bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds())):
            carbon_atoms.append(atom.GetIdx())

    # 3. 构建邻接表（仅包含单键连接的碳原子）
    adj = {i: [] for i in carbon_atoms}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # 仅处理单键连接的碳原子对
        if (bond.GetBondType() == Chem.BondType.SINGLE and
                a1 in carbon_atoms and
                a2 in carbon_atoms):
            adj[a1].append(a2)
            adj[a2].append(a1)

    # 4. 寻找连通分量（BFS遍历）并生成邻接矩阵
    visited = set()
    chains = []
    chains_adjacency = []

    for atom in carbon_atoms:
        if atom not in visited:
            queue = deque([atom])
            visited.add(atom)
            current_chain = []
            chain_adj = {}  # 记录当前碳链中节点间的连接关系

            while queue:
                current = queue.popleft()
                current_chain.append(current)

                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    # 记录当前碳链中的连接关系
                    chain_adj.setdefault(current, []).append(neighbor)
                    chain_adj.setdefault(neighbor, []).append(current)

            # 如果当前碳链长度 ≥ 2，则生成邻接矩阵
            if len(current_chain) >= 2:
                # 将碳链中的原子索引映射到连续索引
                index_map = {idx: i for i, idx in enumerate(sorted(current_chain))}
                n = len(current_chain)
                adjacency_matrix = np.zeros((n, n), dtype=int)

                # 填充邻接矩阵
                for node, neighbors in chain_adj.items():
                    if node in index_map:
                        i = index_map[node]
                        for neighbor in neighbors:
                            if neighbor in index_map:
                                j = index_map[neighbor]
                                adjacency_matrix[i, j] = 1
                                adjacency_matrix[j, i] = 1

                chains.append(sorted(current_chain))
                chains_adjacency.append(adjacency_matrix)

    return chains, chains_adjacency

# 修改fgs官能团
def reset_fgs_carbon(fgs, fgs_adjacency,carbon_chains, carbon_chains_adjacency, marks):
    for index_c,carbon_chain in enumerate(carbon_chains[:]):
        fg_set = get_unique(fgs)
        # 对于长度为2的碳链
        if len(carbon_chain) == 2:
            C1 = carbon_chain[0]
            C2 = carbon_chain[1]

            # 如果是功能C原子，则不管
            if C1 in marks or C2 in marks:
                carbon_chains.remove(carbon_chain)
                continue

            # 两个都存在官能团中
            if C1 in fg_set and C2 in fg_set:
                carbon_chains.remove(carbon_chain)
                continue
            # 如果一个在一个不在
            for i, fg in enumerate(fgs):
                if C1 in fg and C2 not in fg:
                    fgs[i].remove(C1)
                    break
                if C1 not in fg and C2 in fg:
                    fgs[i].remove(C2)
                    break

        # 对于长度大于2的碳链
        if len(carbon_chain) > 2:
            for carbon in carbon_chain[:]:
                # 如果是功能C原子，则不管,
                if carbon in marks:
                    carbon_chain.remove(carbon)
                    continue

                # 如果C在官能团中，找到官能团
                for i, fg in enumerate(fgs):
                    # 如果carbon的删除能保持原本motif的连通性 + 并且要保留自身的联通性，不关键：删除官能团中的碳，
                    if carbon in fg and is_connected_after_removal(fg, fgs_adjacency[i], carbon):
                        # 并且要保留自身的联通性
                        fgs[i].remove(carbon)
                        break
                    # 如果carbon的删除能保持原本motif的连通性，很关键：删除碳链中的碳
                    elif carbon in fg and not is_connected_after_removal(fg, fgs_adjacency[i], carbon):
                        if is_connected_after_removal(carbon_chain,carbon_chains_adjacency[index_c],carbon):
                            carbon_chain.remove(carbon)
                            break
    return fgs, carbon_chains


def get_unique(iterable):
    iterable_set = set()
    for iter in iterable:
        iterable_set.update(iter)
    return iterable_set

def remove_subsets_ring(rings):
    """
    删除集合列表中属于其他集合子集的集合。

    参数:
        sets (list of set): 包含多个集合的列表。

    返回:
        list of set: 去除子集后的集合列表。
    """
    # 创建一个新的列表来存储结果
    result = []

    # 遍历每个集合
    for current_set in rings:
        # 检查当前集合是否是其他集合的子集
        if not any(current_set.issubset(other_set) and current_set != other_set for other_set in rings):
            result.append(current_set)

    return result

def mol_get_motif(mol):
    # 获取芳香键 和 非芳香键
    aromatic_rings, non_aromatic_rings, svg = visualize_ring_aromaticity(mol)
    # print(aromatic_rings)
    # print(non_aromatic_rings)
    # 非芳香键的合并问题
    merged_non_aromatic_rings = merge_aromatic_rings(non_aromatic_rings)
    # print(merged_non_aromatic_rings)
    updated_non_aromatic_rings = merge_single_h_neighbors(mol, merged_non_aromatic_rings)
    rings = updated_non_aromatic_rings + [set(i) for i in aromatic_rings]
    rings = remove_subsets_ring(rings)
    rings_set = set()
    for ring in rings:
        rings_set.update(ring)

    # print("环结构：",rings)
    # 1.标记功能原子
    marks = mark_functional_groups(mol, rings_set)
    # print(marks)

    # 2.合并功能基团及其邻接碳
    fgs, fgs_adjacency = merge_functional_groups(mol, marks, rings_set)
    # print("修改前：",fgs)


    # 3.处理纯碳链
    carbon_chains, carbon_chains_adjacency = find_non_ring_single_bond_only_carbon_chains_with_adjacency(mol)
    # print("碳链：",carbon_chains)

    # 4.修改fgs
    fgs, carbon_chains = reset_fgs_carbon(fgs, fgs_adjacency,carbon_chains, carbon_chains_adjacency, marks)
    carbon_chains = [set(carbon_chains) for carbon_chains in carbon_chains]
    # motifs = process_carbon_chains(mol, fgs, rings_set)

    # 4.处理单独的碳原子
    motifs_result = rings + fgs + carbon_chains # （non-cycle2,环1,碳链3）
    # print("motifs_result",motifs_result)

    # 关于类型的字典：
    # motifs_type_dict = {}
    # for i,ring in enumerate(rings):
    #     if i==0:
    #         motifs_type_dict[0] = list()
    #     motifs_type_dict[0].append(set(ring))
    #
    # for i,fg in enumerate(fgs):
    #     if i==0:
    #         motifs_type_dict[1] = list()
    #     motifs_type_dict[1].append(set(fg))
    #
    # for i,carbon_chain in enumerate(carbon_chains):
    #     if i==0:
    #         motifs_type_dict[2] = list()
    #     motifs_type_dict[2].append(set(carbon_chain))

    # 返回类型列表


    # 碳原子（甲基4）
    i=0
    motifs_list = get_unique(motifs_result)
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        if atom_id not in motifs_list:
            motifs_result.append(set([atom_id]))



    motifs_type = list()
    for motif in motifs_result:
        if motif in rings:
            motifs_type.append(0)
        elif motif in fgs:
            motifs_type.append(1)
        elif motif in carbon_chains:
            motifs_type.append(2)
        else:
            motifs_type.append(3)

    motifs_result = [list(motif) for motif in motifs_result]

    return motifs_type,motifs_result


def get_motif_smiles(mol, motifs_result):

    motif_smiles_list = []

    for motif_indices in motifs_result:
        # 确保索引是从 0 开始（RDKit 原子索引从 0 开始）
        adjusted_indices = [idx - 1 if idx > 0 else idx for idx in motif_indices]  # 调整索引（如果需要）

        try:
            # 使用 MolFragmentToSmiles 生成子结构的 SMILES
            motif_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=adjusted_indices, isomericSmiles=True)
            motif_smiles_list.append(motif_smiles)
        except Exception as e:
            print(f"Error generating SMILES for motif {motif_indices}: {e}")
            motif_smiles_list.append(None)

    return motif_smiles_list


if __name__ == "__main__":
    # 读取数据
    with open('data/ZINC15/zinc15_250k.txt') as f:
        smiles_list = f.read().splitlines()[:1000]
    # datasets = MoleculeNet(root="../dataset/", name="Tox21")
    # dataset = MoleculeNet(root="../dataset/", name="BBBP")
    # smiles = 'Cc1occc1C(=O)Nc2ccccc2'
    # mol = Chem.MolFromSmiles(smiles)
    # motifs_type, motifs_result = mol_get_motif(mol)
    # print(motifs_type)
    # print(motifs_result)
    # motifs_type, motifs_result = mol_get_motif(mol)
    # data =
    # smiles = 'Cc1occc1C(=O)Nc2ccccc2'
    # mol = Chem.MolFromSmiles(smiles)
    # motifs_type, motifs_result = mol_get_motif(mol)
    # svg = visualize_motif(mol, motifs_result,method="save")
    #
    # with open(f'./Image/text.png', 'wb') as file:
    #     file.write(svg)
    #
    # print(motifs_result)

    for i in range(1000):
        mol = Chem.MolFromSmiles(smiles_list[i])
        motifs_type, motifs_result = mol_get_motif(mol)
        motif_smiles = get_motif_smiles(mol, motifs_result)

    # svg = visualize_motif(mol, motifs_result, method="save")
    # with open('./Image/new_motif/temp.png', 'wb') as file:
    #     file.write(svg)
    #     print("temp-保存完毕")
    #
    # print(motifs_result,motif_smiles)
    # for i, smiles in enumerate(smiles_list):
    #
    #     mol = Chem.MolFromSmiles(smiles)
    #
    #     motifs_type,motifs_result = mol_get_motif(mol)
    #     svg = visualize_motif(mol, motifs_result,method="save")
    #
    #     # 保存svg到本地
    #     with open(f'./Image/new_motif/{smiles}.png', 'wb') as file:
    #         file.write(svg)
    #         print(f"{smiles}-保存完毕")

