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


if __name__ == '__main__':
    main()
