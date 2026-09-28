#!/usr/bin/env python3
"""
量子比特 2D 布局图生成器
输入: 量子比特数 n
输出: 2D布局图、耦合对、4色边染色、PNG图片
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict, Counter


# =============================================================================
# 配色方案 (4色高对比度)
# =============================================================================
COLORS = {
    'A': '#E67E22',  # 橙色
    'B': '#C0392B',  # 橙红
    'C': '#27AE60',  # 绿色
    'D': '#2980B9',  # 蓝色
}


# =============================================================================
# 1. 计算 2D 错位布局（4 邻居连线数最多）
# =============================================================================
def E_start_0(a, b):
    """上行 offset 0 (偶数行) → 下行 offset 0.5 (奇数行)"""
    return min(a, b) + min(a, b + 1) - 1


def E_start_half(a, b):
    """上行 offset 0.5 (奇数行) → 下行 offset 0 (偶数行)"""
    return min(a, b) + min(a, b - 1)


def total_edges(d):
    """d 是行大小列表，扫描相邻对按 r 奇偶选公式"""
    total = 0
    for r in range(len(d) - 1):
        a, b = d[r], d[r + 1]
        if r % 2 == 0:
            total += E_start_0(a, b)
        else:
            total += E_start_half(a, b)
    return total


def enumerate_multisets(rows, n, prev=2):
    """枚举多重集（每行 qubit 数的组合）"""
    if rows == 1:
        if n >= prev:
            yield [n]
        return
    for first in range(prev, n // rows + 1):
        for rest in enumerate_multisets(rows - 1, n - first, first):
            yield [first] + rest


def multiset_to_layout(multiset):
    """多重集转 qubit 编号布局"""
    layout = []
    qubit_id = 0
    for size in multiset:
        row = list(range(qubit_id, qubit_id + size))
        layout.append(row)
        qubit_id += size
    return layout


def compute_layout(n):
    # 1. 接收 n (int): 量子比特数
    # 2. 返回 layout (List[List[int]]): 2D 错位布局，4 邻居连线数最多

    if n <= 0:
        return []

    # 小 n 的特殊情况处理
    if n <= 3:
        # n=1: [[0]]
        # n=2: [[0,1]]
        # n=3: [[0,1,2]]
        return [list(range(n))]

    best = None
    best_edges = -1

    # 枚举所有可能的行数
    for rows in range(1, n // 2 + 1):
        for multiset in enumerate_multisets(rows, n):
            # 升序
            edges_asc = total_edges(multiset)
            if edges_asc > best_edges:
                best_edges = edges_asc
                best = multiset

            # 降序（反转）
            desc = list(reversed(multiset))
            edges_desc = total_edges(desc)
            if edges_desc > best_edges:
                best_edges = edges_desc
                best = desc

    return multiset_to_layout(best)


# =============================================================================
# 2. 计算耦合对（4邻居：左下/左上/右上/右下）
# =============================================================================
def compute_couplings(layout):
    # 1. 接收 layout (List[List[int]]): 2D 布局
    # 2. 返回 couplings (List[Tuple[int, int]]): 耦合对列表

    if not layout:
        return []

    rows = len(layout)

    # 构建 qubit -> (row, col) 映射
    qubit_pos = {}
    for r in range(rows):
        row = layout[r]
        # 奇数行偏移 0.5
        offset = 0.5 if r % 2 == 1 else 0
        for c, q in enumerate(row):
            qubit_pos[q] = (r, c + offset)

    couplings = set()

    # 对每对 qubit，如果 row 差 = 1，col 差 = 0 或 1，则连接
    all_qubits = sorted(qubit_pos.keys())
    for i, q1 in enumerate(all_qubits):
        for q2 in all_qubits[i+1:]:
            r1, c1 = qubit_pos[q1]
            r2, c2 = qubit_pos[q2]

            row_diff = abs(r2 - r1)
            col_diff = abs(c2 - c1)

            # row 差 = 1，col 差 = 0 或 1
            if row_diff == 1 and col_diff <= 1:
                couplings.add((min(q1, q2), max(q1, q2)))

    return sorted(list(couplings))


# =============================================================================
# 3. 4 色边染色（每节点 4 边不同色）
# =============================================================================
def edge_coloring(layout, couplings):
    # 1. 接收 layout (List[List[int]]): 2D 布局
    # 2. 接收 couplings (List[Tuple[int, int]]): 耦合对列表
    # 3. 返回 color_assignment (Dict[str, List[Tuple[int, int]]]): 每种颜色包含的耦合对

    colors = ['A', 'B', 'C', 'D']

    # 构建邻接表：每个 qubit 相连的边
    edge_at_node = defaultdict(list)  # qubit -> list of edge indices
    for idx, (q1, q2) in enumerate(couplings):
        edge_at_node[q1].append(idx)
        edge_at_node[q2].append(idx)

    # 按节点的边数从多到少排序（内部节点优先）
    nodes_by_degree = sorted(edge_at_node.keys(), key=lambda q: -len(edge_at_node[q]))

    # 初始化边的颜色
    edge_colors = [None] * len(couplings)

    def get_used_colors_at_node(q):
        """获取节点 q 上已染色的边使用的颜色"""
        used = set()
        for edge_idx in edge_at_node[q]:
            if edge_colors[edge_idx] is not None:
                used.add(edge_colors[edge_idx])
        return used

    # 贪心染色
    for edge_idx, (q1, q2) in enumerate(couplings):
        used1 = get_used_colors_at_node(q1)
        used2 = get_used_colors_at_node(q2)
        used = used1.union(used2)

        # 找第一个可用颜色
        for color in colors:
            if color not in used:
                edge_colors[edge_idx] = color
                break

    # 回溯修复：如果有冲突，尝试调整
    def has_conflict():
        for q, edge_indices in edge_at_node.items():
            used_colors = [edge_colors[i] for i in edge_indices if edge_colors[i] is not None]
            if len(used_colors) != len(set(used_colors)):
                return True
        return False

    # 迭代修复
    max_iterations = 5000
    for _ in range(max_iterations):
        if not has_conflict():
            break

        # 找一个冲突的节点
        for q, edge_indices in edge_at_node.items():
            used_colors = [edge_colors[i] for i in edge_indices if edge_colors[i] is not None]
            if len(used_colors) != len(set(used_colors)):
                # 找到冲突的边，重新分配
                color_count = Counter(used_colors)
                # 找出重复的颜色
                for color, count in color_count.items():
                    if count >= 2:
                        # 找一条用这个颜色的边，尝试换色
                        for ei in edge_indices:
                            if edge_colors[ei] == color:
                                q1, q2 = couplings[ei]
                                used1 = get_used_colors_at_node(q1)
                                used2 = get_used_colors_at_node(q2)
                                used = used1.union(used2)
                                used.discard(color)
                                for c in colors:
                                    if c not in used:
                                        edge_colors[ei] = c
                                        break
                                break
                        break

    # 构建颜色分配
    color_assignment = {c: [] for c in colors}
    for edge_idx, (q1, q2) in enumerate(couplings):
        if edge_colors[edge_idx] is not None:
            color_assignment[edge_colors[edge_idx]].append((q1, q2))

    return color_assignment


def balanced_edge_coloring(couplings):
    """
    均衡 4 色边染色：搜索所有合法染色，目标是最小化 max-min

    参数：
        couplings : List[Tuple[int, int]]，边列表（逻辑编号）
    返回：
        color_assignment : Dict[str, List[Tuple[int, int]]]，均衡染色结果
    """
    from itertools import product

    n_edges = len(couplings)
    colors = ['A', 'B', 'C', 'D']

    # 预计算每条边相连的节点
    edge_nodes = []
    for (q1, q2) in couplings:
        edge_nodes.append((q1, q2))

    best_assignment = None
    best_spread = float('inf')
    best_counts = None

    # 枚举所有 4^n 种染色（n=11 很小）
    for color_assign in product(colors, repeat=n_edges):
        # 检查合法性：同色边无共享端点
        color_to_edges = {c: [] for c in colors}
        for i, c in enumerate(color_assign):
            color_to_edges[c].append(i)

        valid = True
        for c in colors:
            used_nodes = set()
            for ei in color_to_edges[c]:
                n1, n2 = edge_nodes[ei]
                if n1 in used_nodes or n2 in used_nodes:
                    valid = False
                    break
                used_nodes.add(n1)
                used_nodes.add(n2)
            if not valid:
                break

        if not valid:
            continue

        # 计算分布
        counts = [len(color_to_edges[c]) for c in colors]
        spread = max(counts) - min(counts)

        # 更新最优：spread 小优先，再相同按 (A,B,C,D) 字典序大的
        if (spread < best_spread or
            (spread == best_spread and counts > best_counts)):
            best_spread = spread
            best_counts = counts
            best_assignment = {c: [] for c in colors}
            for i, c in enumerate(color_assign):
                best_assignment[c].append(list(edge_nodes[i]))

    # 校验：确保同色无边共享端点（不再要求固定 [3,3,3,2] 分布）
    for c in colors:
        edges = best_assignment[c]
        for i, e1 in enumerate(edges):
            for e2 in edges[i+1:]:
                # 检查是否共享端点
                assert set(e1) & set(e2) == set(), \
                    f"Color {c} has shared endpoint edges: {e1} and {e2}"

    return best_assignment


# =============================================================================
# 4. 绘制布局图
# =============================================================================
def plot_layout(layout, couplings, color_assignment, out_path):
    # 1. 接收 layout (List[List[int]]): 2D 布局
    # 2. 接收 couplings (List[Tuple[int, int]]): 耦合对列表
    # 3. 接收 color_assignment (Dict[str, List[Tuple[int, int]]]): 每种颜色的耦合对
    # 4. 接收 out_path (str): 输出文件路径

    if not layout:
        return

    n = sum(len(row) for row in layout)
    rows = len(layout)
    cols = max(len(row) for row in layout)

    colors = ['A', 'B', 'C', 'D']

    # 根据布局行列数计算 figsize
    # 经验：每个 qubit 占 1.5 英寸宽，1.0 英寸高
    # 顶部留 1.5 英寸放标题，底部留 0.8 英寸放信息框
    # 左侧留 0.8 英寸放行标签，右侧留 1.5 英寸放 legend
    width = max(8, cols * 1.5 + 2.5)  # 至少 8 英寸
    height = max(6, rows * 1.2 + 3.0)  # 至少 6 英寸
    figsize = (width, height)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 构建 qubit 位置映射（考虑错位0.5）
    qubit_pos = {}
    for r in range(rows):
        row = layout[r]
        offset = 0.5 if r % 2 == 1 else 0
        for c, q in enumerate(row):
            qubit_pos[q] = (c + offset, -r)  # y 坐标取负，让第一行在上

    # 构建边的颜色映射
    edge_color_map = {}
    for color, edges in color_assignment.items():
        for e in edges:
            edge_color_map[e] = color
    for e in couplings:
        if e not in edge_color_map:
            edge_color_map[e] = 'A'  # 默认

    # 绘制耦合连线（按颜色，加粗）
    for q1, q2 in couplings:
        if q1 in qubit_pos and q2 in qubit_pos:
            x1, y1 = qubit_pos[q1]
            x2, y2 = qubit_pos[q2]
            color = edge_color_map.get((q1, q2), 'A')
            ax.plot([x1, x2], [y1, y2], color=COLORS[color], linewidth=3.0, alpha=0.85, zorder=1)

    # 绘制 qubit 圆圈（白色填色 + 黑色边框）
    for q, (x, y) in qubit_pos.items():
        circle = plt.Circle((x, y), 0.35, color='white', ec='black', linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, f'q{q}', ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # 设置坐标轴（留边距）
    ax.set_xlim(-1, cols + 0.5)
    ax.set_ylim(-rows - 0.3, 0.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # 行标签
    for r in range(rows):
        ax.text(-0.6, -r, f'Row {r+1}', ha='right', va='center', fontsize=10, fontweight='bold')

    # 图标题
    ax.set_title(f'{n}-qubit {rows}×{cols} staggered coupling ({len(couplings)} pairs, as in Arute Fig. 2)',
                 fontsize=14, fontweight='bold', pad=20)

    # Legend（右上角）
    color_counts = {c: len(color_assignment[c]) for c in colors}
    legend_text = 'Modes\n'
    for c in colors:
        legend_text += f'{c}({color_counts[c]}) '
    ax.text(0.98, 0.98, legend_text.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 底部信息框（用 fig.text，不依赖 ax 范围）
    bottom_text = f'{n} qubits, {len(couplings)} pairs | A:{color_counts["A"]} B:{color_counts["B"]} C:{color_counts["C"]} D:{color_counts["D"]}'
    fig.text(0.5, 0.02, bottom_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='orange'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, facecolor='white')
    plt.close()


# =============================================================================
# 5. 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='量子比特2D布局图生成器')
    parser.add_argument('--n', type=int, required=True, help='量子比特数')
    parser.add_argument('--out', type=str, default=None, help='输出PNG文件路径')
    args = parser.parse_args()

    n = args.n
    out_path = args.out or f'layout_n{n}.png'

    # 计算布局
    layout = compute_layout(n)

    # 计算耦合
    couplings = compute_couplings(layout)

    # 4 色边染色
    color_assignment = edge_coloring(layout, couplings)

    # 统计每种颜色的对数
    color_counts = {c: len(color_assignment[c]) for c in ['A', 'B', 'C', 'D']}

    # 绘图
    plot_layout(layout, couplings, color_assignment, out_path)

    # 输出结果
    result = {
        "n": n,
        "layout": layout,
        "couplings": couplings,
        "color_assignment": color_assignment,
        "color_counts": color_counts,
    }

    print(f"\n{'='*60}")
    print(f"量子比特布局图生成结果 (n={n})")
    print(f"{'='*60}")
    print(f"\n布局: {len(layout)} 行 × {max(len(r) for r in layout)} 列")
    print(f"layout = {layout}")
    print(f"\n耦合对数: {len(couplings)}")
    print(f"couplings = {couplings}")
    print(f"\n4色边染色:")
    for c in ['A', 'B', 'C', 'D']:
        print(f"  {c}: {color_assignment[c]}")
    print(f"\n各颜色对数:")
    print(f"color_counts = {color_counts}")
    print(f"\n图片已保存至: {out_path}")
    print(f"{'='*60}\n")

    return result


# =============================================================================
# 5. 从本源芯片拓扑获取布局（真机模式）
# =============================================================================
# 坏比特黑名单（物理编号）
BAD_QUBITS = {11}  # 实测物理比特11损坏


def find_physical_qubits(n):
    """
    从本源 WK_C180 芯片拓扑中寻找 n 个比特组成的连通子图

    参数：
        n : int，量子比特数
    返回：
        qubits : List[int]，选中的物理比特编号列表
    """
    from uniqc.backend_adapter import find_backend
    from collections import deque

    # 1.获取芯片真实拓扑和可用比特
    backend_info = find_backend('originq:WK_C180')
    topology = backend_info.topology  # List[QubitTopology(u=*, v=*)]
    available_qubits = set(backend_info.extra.get('available_qubits', [])) - BAD_QUBITS
    # 改2: 数据洞过滤——XLSX 中单门保真度<=0 的比特不可选(如 q0)
    import os as _os
    _holes = set()
    for _p in ('../chip_params_WK_C180.json', 'chip_params_WK_C180.json'):
        if _os.path.exists(_p):
            with open(_p, encoding='utf-8') as _f:
                _cp = json.load(_f)
            _holes = {int(q) for q, v in _cp['qubits'].items() if v['gate1q_fidelity'] <= 0}
            break
    available_qubits -= _holes

    # 2.构建邻接表（只考虑可用比特）
    adj = defaultdict(set)
    all_qubits_set = set()
    for edge in topology:
        if edge.u in available_qubits and edge.v in available_qubits:
            adj[edge.u].add(edge.v)
            adj[edge.v].add(edge.u)
            all_qubits_set.add(edge.u)
            all_qubits_set.add(edge.v)

    # 3.BFS从高度数节点出发，找连通子图
    # 按度数降序排序作为起始点
    all_qubits = sorted(all_qubits_set, key=lambda q: -len(adj[q]))

    for start in all_qubits:
        # BFS找连通分量
        visited = {start}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

            if len(visited) >= n:
                # 找到足够的连通比特，按度数降序返回
                result = sorted(visited, key=lambda q: -len(adj[q]))[:n]
                return result

    # 找不到足够的连通比特
    raise ValueError(f"找不到足够的连通比特：需要 {n}")


def generate_abcd_pairs_from_chip(n):
    """
    根据 n 从本源芯片拓扑生成 ABCD 4 批耦合对（真机模式）

    参数：
        n : int，量子比特数
    返回：
        layout : List[List[int]]，2D 布局
        color_assignment : Dict[str, List[Tuple[int, int]]]，每种颜色的耦合对（逻辑编号）
        physical_qubits : List[int]，选中的物理比特编号
    """
    from uniqc.backend_adapter import find_backend

    # 1.从芯片获取 n 个连通的物理比特
    physical_qubits = find_physical_qubits(n)

    # 2.构建物理比特之间的边
    backend_info = find_backend('originq:WK_C180')
    topology = backend_info.topology

    phys_adj = defaultdict(set)
    for edge in topology:
        if edge.u in physical_qubits and edge.v in physical_qubits:
            phys_adj[edge.u].add(edge.v)
            phys_adj[edge.v].add(edge.u)

    # 3.重标号为 0~n-1
    mapping = {phys: logic for logic, phys in enumerate(physical_qubits)}
    couplings = []
    for u in physical_qubits:
        for v in phys_adj[u]:
            if u < v:
                couplings.append((mapping[u], mapping[v]))

    # 4.调用 edge_coloring 染色
    layout = [list(range(n))]
    color_assignment = edge_coloring(layout, couplings)

    return layout, color_assignment, physical_qubits


# =============================================================================
# 6. 从 JSON 选择最优比特（需求单②）
# =============================================================================
import json
from datetime import datetime, timezone
from itertools import combinations


def select_qubits_from_json(json_path, n):
    """
    从 chip_params JSON 中选出 n 个连通、质量最好的比特

    参数：
        json_path : str，JSON 文件路径
        n : int，选取的比特数
    返回：
        dict，包含选中的比特、边、染色等信息
    """
    # 1. 加载 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_qubits = set(int(q) for q in data['qubits'].keys())
    all_edges = {}
    for edge_str, edge_data in data['edges'].items():
        q1, q2 = map(int, edge_str.split('-'))
        all_edges[(q1, q2)] = edge_data['fidelity']

    # 1.5. 剔除度数为0的孤立比特
    degree = defaultdict(int)
    for (q1, q2) in all_edges.keys():
        degree[q1] += 1
        degree[q2] += 1
    # 改2: 数据洞过滤——单门保真度<=0 的比特不可选
    _nhole = sum(1 for q in all_qubits if not data['qubits'][str(q)]['gate1q_fidelity'] > 0)
    print(f"改2: 数据洞比特剔除 {_nhole} 个")
    candidates = {q for q in all_qubits
                  if degree[q] > 0 and data['qubits'][str(q)]['gate1q_fidelity'] > 0}
    print(f"Total qubits: {len(all_qubits)}, isolated (degree=0) removed: {len(all_qubits) - len(candidates)}")

    # 2. 枚举所有 n 比特子集，找连通的
    connected_subsets = []
    all_qubits_list = sorted(candidates)

    for subset in combinations(all_qubits_list, n):
        subset_set = set(subset)

        # 找内部边（两端都在 subset 中）
        internal_edges = []
        for (q1, q2), fid in all_edges.items():
            if q1 in subset_set and q2 in subset_set:
                internal_edges.append(((q1, q2), fid))

        # BFS 判定连通性
        if len(internal_edges) < n - 1:
            continue

        # 构建邻接表
        adj = defaultdict(set)
        for (q1, q2), _ in internal_edges:
            adj[q1].add(q2)
            adj[q2].add(q1)

        # BFS from first qubit
        visited = {subset[0]}
        queue = [subset[0]]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        if len(visited) == n:
            # 连通！计算分数
            # ① 内部边数多的优先
            # ② 边数相同 → e_q 总和小的优先
            # ③ 再相同 → 比特编号列表字典序小的优先
            eq_sum = sum(data['qubits'][str(q)]['e_q'] for q in subset)
            score = (-len(internal_edges), eq_sum, list(subset))
            connected_subsets.append((score, subset, internal_edges))

    if not connected_subsets:
        raise ValueError(f"找不到 {n} 个连通的比特子集")

    # 3. 排序选最佳
    connected_subsets.sort(key=lambda x: x[0])
    best_score, best_subset, best_edges = connected_subsets[0]

    # 4. 4色边染色（复用现有逻辑）
    # 构建边列表和 color_assignment
    edges_list = [(q1, q2) for ((q1, q2), _) in best_edges]
    edge_fidelities = {f"{q1}-{q2}": fid for ((q1, q2), fid) in best_edges}

    # 用逻辑编号（0~n-1）做均衡染色
    mapping = {phys: i for i, phys in enumerate(best_subset)}
    logical_edges = [(mapping[q1], mapping[q2]) for (q1, q2) in edges_list]
    color_assignment = balanced_edge_coloring(logical_edges)

    # 5. 校验染色合法性：每种颜色内无共享比特
    for color, edges in color_assignment.items():
        used_nodes = set()
        for e in edges:
            n1, n2 = e
            assert n1 not in used_nodes, f"Color {color} has shared node {n1}"
            assert n2 not in used_nodes, f"Color {color} has shared node {n2}"
            used_nodes.add(n1)
            used_nodes.add(n2)

    # 6. 边数总和校验
    total_colored = sum(len(edges) for edges in color_assignment.values())
    assert total_colored == len(edges_list), \
        f"Color assignment edge count {total_colored} != internal edge count {len(edges_list)}"

    # 7. 计算统计
    eq_values = [data['qubits'][str(q)]['e_q'] for q in best_subset]
    avg_eq = sum(eq_values) / len(eq_values)
    min_cz_fid = min(fid for _, fid in best_edges)

    # 8. 构建 positions（画图坐标）
    positions = {}
    if 'positions' in data:
        for q in best_subset:
            r, c = data['positions'][str(q)]
            # 错位: 奇数行 offset 0.5
            x = c + 0.5 * (r % 2)
            y = -r
            positions[str(q)] = [round(x, 1), round(y, 1)]

    # 9. 构建输出
    result = {
        "backend": data['backend'],
        "n": n,
        "physical_qubits": list(best_subset),
        "edges": [[q1, q2] for (q1, q2) in edges_list],
        "edge_fidelities": edge_fidelities,
        "avg_eq": avg_eq,
        "min_cz_fidelity": min_cz_fid,
        "color_assignment": {c: list(es) for c, es in color_assignment.items()},
        "selected_at": datetime.now(timezone.utc).isoformat(),
    }

    # 添加 rows 和 positions（如果有）
    if 'rows' in data:
        result['rows'] = data['rows']
    if positions:
        result['positions'] = positions

    return result


def main_select():
    """CLI 入口：select 模式"""
    import argparse
    parser = argparse.ArgumentParser(description='从JSON选择最优比特')
    parser.add_argument('--select', action='store_true', help='启用选择模式')
    parser.add_argument('--json', required=True, help='输入 JSON 文件路径')
    parser.add_argument('--n', type=int, required=True, help='选取比特数')
    parser.add_argument('--out', required=True, help='输出 JSON 文件路径')
    args = parser.parse_args()

    result = select_qubits_from_json(args.json, args.n)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Selected {result['n']} qubits from {args.json}")
    print(f"{'='*60}")
    print(f"Physical qubits: {result['physical_qubits']}")
    print(f"Internal edges: {len(result['edges'])}")
    print(f"Avg e_q: {result['avg_eq']:.4f}")
    print(f"Min CZ fidelity: {result['min_cz_fidelity']:.4f}")
    print(f"\nABCD coloring:")
    for c in ['A', 'B', 'C', 'D']:
        edges = result['color_assignment'][c]
        print(f"  {c}: {edges}")

    # 写入输出
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {args.out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--select':
        main_select()
    else:
        main()
