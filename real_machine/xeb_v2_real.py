# XEB 实验复现（基于 uniqc 代码库）
# 5 步实验流程：
#   步骤 1   ：生成随机线路 U（n qubit × m cycle）
#   步骤 2.1 ：无噪声演化（state vector 模拟器）→ 纯态 |ψ⟩
#   步骤 2.2 ：含噪声演化（statevector 模拟器 + Depolarizing）→ 采样
#   步骤 3   ：采样 N_s 个比特串
#   步骤 4   ：算 F_XEB
#   步骤 5   ：重复 K 次取期望和标准差
#
# 实现所需的功能模块（与步骤对应）：
# 模块 1:量子线路构建 → 步骤 1：生成随机线路 U
# 模块 2:模拟与测量 → 步骤 2.1 + 2.2 + 步骤 3：跑模拟器(无噪声/含噪声演化) + 采样 N_s 个比特串
# 模块 3:保真度计算 → 步骤 4：算 F_XEB
# 模块 4:重复实验 → 步骤 5：重复 K 次取期望和标准差
#
# 实验参数（n 可取任意值，20 cycle / K=10，与 Sycamore 实验一致）：
#   n  = 10~24    qubit 数（通过 --n 或 --n_list 指定）
#   m  = 20        cycle 数（ABCDCDAB × 2 + ABCD = 20 cycle）
#   N_s= 2000     单次实验采样数（10³ × 2 = 2000）
#   K  = 10        独立电路重复数
#
# 典型实验：n=10,12,14,16,18,20,22,24
#
# 噪声参数（来自标定实验 calibrate.py 实测，isolated + Neill 2017 公式）：
#   EPS1       = 0.0016  单量子门 p 值（去极化概率，用于 Depolarizing 演化）
#   EPS2       = 0.0062  双量子门 p 值（去极化概率，用于 TwoQubitDepolarizing 演化）
#   EPS1_ERR   = 0.00172 单量子门实测错误率 ε₁（用于 alpha_f 预测）
#   EPS2_ERR   = 0.00673 双量子门实测错误率 ε₂（用于 alpha_f 预测）
#   EQ_ERR     = 0.00000 读出错误率 e_q（uniqc 无读出错误模型）
#
# 使用方法：
#   将UNIQC_XXX() 函数替换为uniqc库的真实 API。
#   所有占位符函数当前都会 raise NotImplementedError——替换完才能跑。
#
# F_XEB计算公式：
#   F_XEB = 2^n · Σ_i P_sampled(x_i) · P_expected(x_i) - 1
#
# 命令行：
#   python xeb_v2.py --n 10            # n=10，无噪声
#   python xeb_v2.py --n 20 --noise    # n=20，含噪声
#   python xeb_v2.py --n_list 10,12,14 # 批量跑多个 n
#   python xeb_v2.py --n 10 --K 5      # n=10，重复 5 次
#
# 要改噪声大小，直接改源码顶部的 EPS1/EPS2 即可；不要噪声就把 --noise 去掉


# ============================================================
# 实验参数（XEB 标准配置）
# ============================================================
# XEB 标准参数（n=10, m=20, K=10, N_s=2000）
N_QUBITS    = 10       # 默认 qubit 数
M_CYCLES    = 20       # cycle 数（ABCDCDAB × 2 + ABCD = 20）
N_s         = 2000      # 单次实验采样数
K_REPEATS   = 10        # 独立电路重复数
EPS1        = 0.0016    # 单量子门 p 值（用于含噪声模拟）
EPS2        = 0.0062    # 双量子门 p 值（用于含噪声模拟）
EPS1_ERR    = 0.00172   # 单量子门实测错误率 ε₁
EPS2_ERR    = 0.00673   # 双量子门实测错误率 ε₂
EQ_ERR      = 0.00000   # 读出错误率 e_q（uniqc 无读出错误模型）

# ============================================================
# 代码库调用
# ============================================================
import numpy as np
import random
import argparse
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from uniqc import Circuit, Simulator, submit_task, submit_batch, wait_for_result, compile
from uniqc.backend_adapter import find_backend
from uniqc.simulator import Depolarizing, ErrorLoader_GateTypeError, NoisySimulator, ErrorLoader_GenericError, TwoQubitDepolarizing

# ============================================================
# 真机配置（优先从环境变量读取，其次从 uniqc 配置系统读取）
# ============================================================
# 优先从环境变量读取
ORIGINQ_API_KEY = os.environ.get("ORIGINQ_API_KEY", "")

if not ORIGINQ_API_KEY:
    # 尝试从 uniqc 配置系统读取
    try:
        from uniqc.config import load_originq_config
        originq_config = load_originq_config()
        ORIGINQ_API_KEY = originq_config.get("api_key", "")
        ORIGINQ_TOKEN = originq_config.get("token", "")
        # 兼容 token 和 api_key 两种配置
        if not ORIGINQ_API_KEY and ORIGINQ_TOKEN:
            ORIGINQ_API_KEY = ORIGINQ_TOKEN
    except Exception as e:
        ORIGINQ_API_KEY = ""

BACKEND = "originq:WK_C180_2"  # 默认后端
LAYOUT_JSON_PATH = None  # 布局 JSON 路径
CHIP_PARAMS_JSON_PATH = None  # 芯片参数 JSON 路径
LAST_TASK_ID = None  # 最近一次提交的 task_id

print(f"[Real Machine Config]")
print(f"  API Key loaded: {ORIGINQ_API_KEY is not None and len(ORIGINQ_API_KEY) > 0}")
print(f"  Default Backend: {BACKEND}")
if not ORIGINQ_API_KEY:
    print(f"  WARNING: No API key found! Set ORIGINQ_API_KEY env var or run: uniqc config set originq.token <YOUR_TOKEN>")


# 单元测试（已移至 tests.py）
from tests import run_self_test


# 从 layout_gen 导入量子布局相关函数
from layout_gen import compute_layout, compute_couplings, edge_coloring, plot_layout, generate_abcd_pairs_from_chip, select_qubits_from_json

# 导入 JSON 加载函数（新增）
import json
from datetime import datetime, timezone


def load_layout_from_json(json_path, n):
    """从 JSON 文件加载布局，若不存在则自动生成"""
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        print(f"[Layout] Loaded from {json_path}")
        return layout_data
    else:
        # 自动生成
        print(f"[Layout] File not found, auto-generating...")
        # 需要 chip_params JSON
        if CHIP_PARAMS_JSON_PATH is None:
            raise ValueError("--layout JSON not found and --chip-params not specified")
        layout_data = select_qubits_from_json(CHIP_PARAMS_JSON_PATH, n)
        # 保存
        with open(json_path, 'w') as f:
            json.dump(layout_data, f, indent=2)
        print(f"[auto] layout 已生成: {json_path}")
        return layout_data


def load_chip_params(json_path):
    """从 JSON 文件加载芯片参数"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"[Chip Params] Loaded from {json_path}")
    return data


def check_backend_alignment(layout_data, backend):
    """时效对账：检查 backend 与 layout 的 backend 字段是否一致"""
    layout_backend = layout_data.get('backend', '')
    assert layout_backend == backend, \
        f"Backend mismatch: layout backend={layout_backend}, requested={backend}"


def check_topology_alignment(layout_data, backend):
    """时效对账：检查实时 topology 是否覆盖 layout 的所有边和比特"""
    backend_info = find_backend(backend)
    available_qubits = set(backend_info.extra.get('available_qubits', []))
    layout_qubits = set(layout_data['physical_qubits'])

    missing_qubits = layout_qubits - available_qubits
    assert not missing_qubits, \
        f"Qubits in layout not available on backend: {missing_qubits}. Please re-download XLSX and regenerate layout."

    # 检查边：layout_data['edges'] 是逻辑编号，需要映射到物理编号
    topology = backend_info.topology
    available_edges = set()
    for edge in topology:
        available_edges.add((min(edge.u, edge.v), max(edge.u, edge.v)))

    # 将逻辑边转为物理边
    physical_qubits = layout_data['physical_qubits']
    layout_edges = set()
    for e in layout_data['edges']:
        phys1 = physical_qubits[e[0]]
        phys2 = physical_qubits[e[1]]
        layout_edges.add((min(phys1, phys2), max(phys1, phys2)))

    missing_edges = layout_edges - available_edges
    assert not missing_edges, \
        f"Edges in layout not available on backend: {missing_edges}. Please re-download XLSX and regenerate layout."

    print(f"[Topology Check] Qubits: all OK, Edges: all OK")


# 改3: 门时长配置(ns)——本源官方数值确认后只改这里(当前为典型值锚定)
GATE_DUR_NS = {"1q": 20.0, "sx": 20.0, "SX": 20.0, "rz": 0.0, "RZ": 0.0, "cz": 70.0, "CZ": 70.0, "measure": 0.0, "MEASURE": 0.0}


def schedule_total_duration_ns(compiled_circuit, backend_info):
    """改3: 用 uniqc timeline 调度器计算电路总运行时长 T_总 (ns), 全比特共用."""
    from uniqc.visualization.timeline import schedule_circuit
    sch = schedule_circuit(compiled_circuit, backend_info=backend_info,
                           gate_durations=GATE_DUR_NS, unit='ns')
    return float(sch.total_duration)


def check_data_holes(layout_data, chip_params):
    """改2: 数据洞过滤——布局内任何比特单门保真度<=0 即拒绝."""
    qs = chip_params['qubits']
    for q in layout_data['physical_qubits']:
        f1 = qs[str(q)]['gate1q_fidelity']
        if f1 <= 0:
            raise ValueError('[改2] 数据洞比特混入布局: q%s gate1q_fidelity=%s' % (q, f1))
    print('[改2] data-hole check: PASS')


def run_depth_scan(layout_data, chip_params, n, ms, k, seed_base, shots,
                    backend_info, dry_run=False):
    """深度扫描实验（2026-09-23 CLI 收编版）: ms x k 条种子线路, 整批提交.
    逐线路自检: CZ 计数与染色一致 / 零 SWAP / 测量映射.
    返回汇总 dict (dry_run 模式返回 None, 只打印自检与 alpha)."""
    phys = layout_data['physical_qubits']
    ca = layout_data['color_assignment']
    jobs = []
    for m in ms:
        cc = {}
        for i in range(m):
            cc[PATTERN_8CYCLE[i % 8]] = cc.get(PATTERN_8CYCLE[i % 8], 0) + 1
        expected_cz = sum(len(ca[c]) * cc[c] for c in ca)
        for kk in range(k):
            seed = seed_base + 100 * m + kk
            circ = generate_random_circuit(n, m, ca, seed=seed)
            comp = compile(circ, backend_info=backend_info, available_qubits=phys)
            ncz = sum(1 for line in comp.originir.split('\n')
                      if line.strip().startswith('CZ'))
            assert ncz == expected_cz, 'CZ mismatch m=%s seed=%s: %s vs %s' % (m, seed, ncz, expected_cz)
            nswap = sum(1 for line in comp.originir.split('\n')
                        if line.strip().startswith('SWAP'))
            assert nswap == 0, 'SWAP appeared m=%s seed=%s' % (m, seed)
            psi = Simulator('statevector').simulate_statevector(circ)
            norm = {}
            for line in comp.originir.split('\n'):
                line = line.strip()
                if line.startswith('MEASURE'):
                    parts = line[7:].strip().split(',')
                    q = int(parts[0].split('[')[1].split(']')[0])
                    cb = int(parts[1].split('[')[1].split(']')[0])
                    norm[cb] = phys.index(q) if q in phys else q
            t_total = schedule_total_duration_ns(comp, backend_info)
            alpha = compute_alpha_f_exact(layout_data, chip_params, n, m, ca, t_total_ns=t_total)[0]
            jobs.append({'m': m, 'seed': seed, 'psi': psi, 'norm': norm, 'compiled': comp,
                         'alpha': alpha, 't_total': t_total})
        print('m=%s: %d circuits built (expected_cz=%d)' % (m, k, expected_cz), flush=True)
    print('total %d circuits built' % len(jobs), flush=True)

    if dry_run:
        for m in ms:
            alphas = [j['alpha'] for j in jobs if j['m'] == m]
            ts = [j['t_total'] for j in jobs if j['m'] == m]
            print('m=%s: alpha_mean=%.4f T_total=%.0f~%.0fns' %
                  (m, float(np.mean(alphas)), min(ts), max(ts)), flush=True)
        print('[DRY-RUN] checks passed, NOT submitting.', flush=True)
        return None

    task_id = submit_batch([j['compiled'] for j in jobs], backend=BACKEND, shots=shots)
    print('batch task: %s' % task_id, flush=True)
    print('waiting for results (timeout 10800s)...', flush=True)
    res = wait_for_result(task_id, timeout=21600)
    print('results received', flush=True)
    items = res if isinstance(res, list) else (res.results if hasattr(res, 'results') else res.get('results'))
    assert len(items) == len(jobs), 'item count mismatch %d vs %d' % (len(items), len(jobs))

    rows = []
    for j, it in zip(jobs, items):
        counts = it.counts if hasattr(it, 'counts') else it.get('counts')
        bits = []
        for kk_, cnt in counts.items():
            s = str(kk_).strip()
            if s[:2].lower() == '0b':
                s = s[2:]
            cidx = int(s, 2)
            logical = 0
            for b in range(n):
                if (cidx >> b) & 1:
                    logical |= (1 << j['norm'].get(b, b))
            bits.extend([logical] * cnt)
        F = compute_F_XEB(j['psi'], bits, n)
        rows.append({'m': j['m'], 'seed': j['seed'], 'F': round(F, 4),
                     'alpha': round(j['alpha'], 4), 't_total': round(j['t_total'], 1)})
        print('m=%s seed=%s F=%.4f' % (j['m'], j['seed'], F), flush=True)

    print('===== 汇总: n=%d 深度扫描 (含④与无④) =====' % n, flush=True)
    print('| m | F 均值±SEM | α含④ | 比值 | σ偏离 | α无④ | 比值 | σ偏离 |', flush=True)
    print('|---|---|---|---|---|---|---|---|', flush=True)
    summary = []
    for m in ms:
        fs = np.array([r['F'] for r in rows if r['m'] == m])
        al = np.array([r['alpha'] for r in rows if r['m'] == m])
        fm = fs.mean()
        se = fs.std(ddof=1) / np.sqrt(len(fs))
        a4 = al.mean()
        _, eq, g1, cz, dc = compute_alpha_f_exact(layout_data, chip_params, n, m, ca, t_total_ns=None)
        ano = eq * g1 * cz
        r4 = ('%.2f' % (fm / a4)) if (a4 > 1e-4 and fm > 0) else '—'
        rn = ('%.2f' % (fm / ano)) if fm > 0 else '—'
        s4 = (fm - a4) / se
        sn = (fm - ano) / se
        summary.append({'m': m, 'K': len(fs), 'F_mean': round(fm, 4), 'F_SEM': round(se, 4),
                        'alpha_mean': round(a4, 4), 'alpha_no4': round(ano, 4)})
        print('| %d | %.4f±%.4f | %.4f | %s | %+.1fσ | %.4f | %s | %+.1fσ |' %
              (m, fm, se, a4, r4, s4, ano, rn, sn), flush=True)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_file = 'depthscan_n%d_k%d_%s.json' % (n, k, stamp)
    json.dump({'backend': BACKEND, 'n': n, 'K': k, 'ms': ms, 'seed_base': seed_base,
               'shots': shots, 'cluster': phys, 'rows': rows, 'summary': summary},
              open(out_file, 'w'), indent=1)
    print('saved', out_file, flush=True)
    print('ALL DONE', flush=True)
    return None


def compute_alpha_f_exact(layout_data, chip_params, n, m, color_assignment, t_total_ns=None):
    """
    精确版 alpha_f 计算（2026-09-23 改1 新口径）
    alpha_f = ∏(1−e_q) × ∏F1q_q^m × ∏F_CZ_e^{n_e} × ∏_q (2·e^(−T/T2*_q) + e^(−T/T1_q)) / 3

    - 门保真度按"存活率"直接连乘, 无 2x/4/3 换算（口径 B, 用户拍板）
    - 退相干项用电路总运行时长 T_总(ns), 覆盖干活+空闲全程（改3）
    """
    import math
    physical_qubits = layout_data['physical_qubits']
    qubits_params = chip_params['qubits']
    edges_fid = chip_params['edges']

    # 1. 读出项 ∏(1−e_q): 每比特乘 1 次
    eq_product = 1.0
    for q in physical_qubits:
        eq = qubits_params[str(q)]['e_q']
        eq_product *= (1 - eq)

    # 2. 单门项 ∏ F1q^m: 每比特乘 m 次, 保真度直接连乘
    gate1q_product = 1.0
    for q in physical_qubits:
        f1 = qubits_params[str(q)]['gate1q_fidelity']
        if f1 <= 0:
            raise ValueError('[改2] 数据洞比特: q%s gate1q_fidelity=%s' % (q, f1))
        gate1q_product *= f1 ** m

    # 3. 双门项 ∏ F_CZ^{n_e}: 每条边乘它被拍的次数(按色计数), 保真度直接连乘
    pattern = ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B']
    color_counts = [0, 0, 0, 0]
    for i in range(m):
        color_idx = ['A', 'B', 'C', 'D'].index(pattern[i % 8])
        color_counts[color_idx] += 1

    cz_product = 1.0
    for edge in layout_data['edges']:
        lq1, lq2 = edge[0], edge[1]
        pq1, pq2 = physical_qubits[lq1], physical_qubits[lq2]
        edge_key = '%d-%d' % (min(pq1, pq2), max(pq1, pq2))
        edge_fid = edges_fid.get(edge_key, {}).get('fidelity')
        if edge_fid is None:
            raise ValueError('边 %s 不在标定表中' % edge_key)
        times = 0
        for c_idx, cname in enumerate(['A', 'B', 'C', 'D']):
            for pair in color_assignment.get(cname, []):
                if pair == [lq1, lq2] or pair == [lq2, lq1]:
                    times += color_counts[c_idx]
        cz_product *= edge_fid ** times

    # 4. 退相干项: T_总(ns) 转 us, 权重 相位(T2*) 2/3 + 弛豫(T1) 1/3
    deco_product = 1.0
    if t_total_ns is not None:
        t_us = t_total_ns / 1000.0
        for pq in physical_qubits:
            t1 = qubits_params[str(pq)].get('t1_us', 35.0)
            t2s = qubits_params[str(pq)].get('t2star_us', 4.5)
            deco_product *= (2 * math.exp(-t_us / t2s) + math.exp(-t_us / t1)) / 3

    alpha_f_exact = eq_product * gate1q_product * cz_product * deco_product
    return alpha_f_exact, eq_product, gate1q_product, cz_product, deco_product



# ============================================================
# 参数预设：单门集合与公共参数
# ============================================================
# 门名以 uniqc 文档为准——先查 uniqc 文档确认这三个门叫什么。
# 可能的名字（按常见约定猜测，仅供参考）：
#   √X = 'sx' / 'sqrtX' / 'X90'
#   √Y = 'sy' / 'sqrtY' / 'Y90'
#   √W = 'sw' / 'sqrtW' / 'XY90'（其中 W=(X+Y)/√2）
# 填错时 simulate 不报错但结果不对——跑无噪声版本 F≈1 即可确认。
SINGLE_GATES = ['sqrtX', 'sqrtY', 'sqrtW']  # √X, √Y, √W，其中 W=(X+Y)/√2

PATTERN_8CYCLE = ['A', 'B', 'C', 'D', 'C', 'D', 'A', 'B']  # 8 cycle 完整序列


# ============================================================
# 参数预设： 双量子比特拓扑（四种模式的双门配对的子集）
# ============================================================
#20 qubit（4×5 错位布局），共 27 对双门：
# 8 cycle 序列 ABCDCDAB
#
# qubit 编号：q0=0, q1=1, …, q19=19
# 图3.1 布局：4 行 × 5 列错位——
#   行 1: q0 q2 q4 q6 q8
#   行 2: q1 q3 q5 q7 q9
#   行 3: q10 q12 q14 q16 q18
#   行 4: q11 q13 q15 q17 q19
ABCD_PAIRS = {
    # A 红色：
    'A': [(1, 12), (3, 14), (4, 5), (6, 7), (8, 9), (15, 16), (17, 18)],
    # B 橙色：
    'B': [(3, 12), (5, 14), (0, 1), (7, 8), (9, 18), (10, 11), (16, 17)],
    # C 绿色：
    'C': [(7, 16), (1, 2), (3, 4), (5, 6), (11, 12), (13, 14), (18, 19)],
    # D 蓝色：
    'D': [(5, 16), (7, 18), (2, 3), (1, 10), (12, 13), (14, 15)],
}


# ============================================================
# 【模块 0：量子布局生成】调用 layout_gen 生成 ABCD 4 批耦合对
# ============================================================
def generate_abcd_pairs(n):
    """
    根据 n 调用 layout_gen，自动生成 ABCD 4 批耦合对

    参数：
        n : int，量子比特数
    返回：
        layout : List[List[int]]，2D 错位布局
        color_assignment : Dict[str, List[Tuple[int, int]]]，每种颜色(A/B/C/D)的耦合对
    """
    layout = compute_layout(n)
    couplings = compute_couplings(layout)
    color_assignment = edge_coloring(layout, couplings)
    return layout, color_assignment

# ============================================================
# 【模块 1：量子线路构建】步骤 1 — 生成随机线路 U
# ============================================================
# 函数功能：生成 n qubit × m cycle 的随机线路 U
# ============================================================
# 参数说明：
#   n               : 比特数
#   m               : cycle 数
#   color_assignment: 可选参数，Dict[str, List[Tuple[int, int]]]，从 layout_gen 生成的 4 色耦合对
#                     若为 None，则使用默认的 ABCD_PAIRS 字典
#   seed            : 随机种子（保证线路可复现，即同 seed=相同量子线路；内部作为 random.seed() 的起点）
#   circuit         : 返回值，生成的随机量子线路对象（uniqc 线路）
# ============================================================
def generate_random_circuit(n, m, color_assignment, seed=None):
    # 强制要求传入 color_assignment，不使用硬编码 ABCD_PAIRS
    assert color_assignment is not None, \
        "color_assignment 不能为 None，否则会用 n=20 硬编码 ABCD_PAIRS"
    # 使用 layout_gen 生成的 4 色耦合对
    pairs_dict = color_assignment

    # 1.设置随机种子seed,即随机函数random的起点，用以保证实验可复现同 seed=生成相同电路，不同 seed=不同电路（生成K=10种不同线路，分别测量F_XEB并取均值）
    if seed is not None:
        random.seed(seed)
    # 2.调用 uniqc 库接口，创建n qubit量子线路对象
    circuit = Circuit(n)
    # 3.生成一个长度为 n 的列表 last_single_gate（初值全 None），用于记录每个 qubit 上一 cycle 用的单门（满足论文约束：同一 qubit 相邻 cycle 单门不得连续重复）
    last_single_gate = [None]*n
    # 4.使用循环生成完整量子线路：遍历 m 个 cycle 逐层构造线路，每个 cycle 内先施加单量子门层、再施加双量子门层
    for cycle in range(m):
        #   4.1 单量子门层：对每个 qubit 从单门集合{√X, √Y, √W}中随机选 1 个施加，且不与该 qubit 上一 cycle 用过的门重复（检查第3步记录的列表）
        for qubit in range(n):
            #       4.1.1 生成候选集：从单门集合{√X, √Y, √W}中剔除"该 qubit 上一 cycle 用过的门"（检查第3步记录的列表）
            if last_single_gate[qubit] is None:
                candidates = SINGLE_GATES[:]
            else:
                candidates = [g for g in SINGLE_GATES if g != last_single_gate[qubit]]
            #       4.1.2 选门：从候选集中随机抽 1 个单门
            gate  = random.choice(candidates)
            # 4.1.3 施加单门：用 RPhi 实现 √X, √Y, √W（θ=π/2 固定）
            #   RPhi(θ=π/2, φ=0)     等价 √X
            #   RPhi(θ=π/2, φ=π/2)   等价 √Y
            #   RPhi(θ=π/2, φ=π/4)   等价 √W
            if gate == 'sqrtX':
                circuit.rphi(qubit, theta=np.pi/2, phi=0)
            elif gate == 'sqrtY':
                circuit.rphi(qubit, theta=np.pi/2, phi=np.pi/2)
            elif gate == 'sqrtW':
                circuit.rphi(qubit, theta=np.pi/2, phi=np.pi/4)
            # 4.1.4 记录该cycle使用的单门：更新该 qubit 的"上一 cycle 单门"记录到列表，供下一 cycle 参考
            last_single_gate[qubit] = gate
    #   4.2 双量子门层：按 ABCDCDAB 序列查本 cycle 应使用的子集，遍历子集内所有 qubit 对施加双门
        #   4.2.1 确定本序列字母：将cycle序号对8取模，查 ABCDCDAB 序列得本 cycle 子集对应的字母（A/B/C/D）
        letter = PATTERN_8CYCLE[cycle % 8]
        #   4.2.2 查双量子比特对：查 pairs_dict（color_assignment 或 ABCD_PAIRS）得本 cycle 要施加的所有 qubit 对
        pairs = pairs_dict[letter]
        #   4.2.3 施加双门：用 CZ（按 color_assignment 颜色 pattern 打）
        for (q1, q2) in pairs:
            if q1 < n and q2 < n:
                circuit.cz(q1, q2)
    # 5.在线路的每个 qubit 上添加测量门
    for i in range(n):
        circuit.measure(i)
    # 6.返回生成的量子线路对象 circuit
    return circuit


# ============================================================
# 【模块 2：模拟与测量】步骤 2.1 + 2.2 + 步骤 3 — 跑模拟器 + 采样 N_s 个比特串
# ============================================================
# 函数功能：
#   步骤 2.1：无噪声演化 → 纯态 |ψ⟩
#   步骤 2.2：含噪声演化 → 采样（模拟真实测量）
#   步骤 3：采样 N_s 个比特串
# ============================================================
# 参数说明：
#   circuit    : 量子线路对象（uniqc Circuit）
#   use_noise : 是否使用噪声（True=含噪声，False=无噪声）
#   N_s       : 采样次数 N_s
#   seed      : 随机种子（用于手动采样时的随机数生成）
# 返回值：
#   bitstrings : 采样的比特串列表（整数形式，如 [0, 3, 7, ...]）
#   psi       : 无噪声模拟得到的纯态矢量 |ψ⟩（复数 numpy 数组）
# ============================================================
def run_circuit(circuit, use_noise, N_s, seed=None, progress_callback=None):
    # 1.创建无噪声模拟器（用于获取理想态|ψ⟩）
    sim_ideal = Simulator('statevector')
    # 2.若 use_noise=True，则定义噪声模型并创建含噪声模拟器
    if use_noise:
        # 2.1 定义噪声模型
        # 注意：generic_error=[Depolarizing(p=0)] 不对任何门默认加噪声
        # 单门噪声通过 gatetype_error 专门加到 U3 门上
        # 双门噪声加到 ISWAP 门上
        error_model = ErrorLoader_GateTypeError(
            generic_error=[Depolarizing(p=0)],  # 无默认噪声
            gatetype_error={
                'RPhi': [Depolarizing(p=EPS1)],        # 单门噪声
                'CZ': [TwoQubitDepolarizing(p=EPS2)],   # 双门噪声
            }
        )
        # 2.2 创建含噪声模拟器
        sim_noisy  = NoisySimulator(backend_type='statevector', error_loader=error_model)
    # 3.用无噪声模拟器跑线路，得到理想的纯态矢量|ψ⟩
    psi = sim_ideal.simulate_statevector(circuit)
    # 4.根据模拟器类型选择测量方式得到比特串
    #   4.1 若 use_noise=False：手动按 psi 概率分布采样（uniqc simulate_shots 有 bug，不按概率分布）
    if not use_noise:
        # uniqc 的 simulate_shots 对复杂线路返回均匀分布，这里手动按 psi 概率采样
        p = np.abs(psi) ** 2
        p = p / p.sum()  # 归一化
        bitstrings = np.random.choice(len(p), size=N_s, p=p).tolist()
    #   4.2 若 use_noise=True：含噪声模拟器分批采样 N_s 次
    else:
        bitstrings = []
        batch_size = 100
        n_batches = (N_s + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            current_batch = min(batch_size, N_s - len(bitstrings))
            shot_result = sim_noisy.simulate_shots(circuit.originir, shots=current_batch)
            for bitstring, count in shot_result.items():
                bitstrings.extend([bitstring] * count)
            # 调用进度回调
            if progress_callback is not None:
                progress_callback(len(bitstrings), N_s, psi, bitstrings, circuit.qubit_num)
    # 5.返回比特串列表和纯态矢量|ψ⟩
    return bitstrings, psi


# ============================================================
# 【模块 2.5：真机运行】提交到 OriginQ 真机并获取结果
# ============================================================
def run_circuit_on_real_machine(circuit, n, N_s, physical_qubits=None, color_assignment=None):
    """
    提交量子线路到 OriginQ 真机并获取测量结果

    参数：
        circuit : uniqc Circuit 对象
        n       : qubit 数
        N_s     : 采样次数
        physical_qubits : List[int]，选中的物理比特编号（用于限死比特池）
        color_assignment : Dict[str, List[Tuple[int, int]]]，ABCD 耦合对（用于验证 CZ 数量）

    返回：
        bitstrings : 采样的比特串列表（整数形式）
    """
    if ORIGINQ_API_KEY is None or len(ORIGINQ_API_KEY) == 0:
        raise ValueError("ORIGINQ_API_KEY 环境变量未设置！")

    # 1. 先编译电路到目标后端（获取物理比特映射和实际门数）
    print(f"[Real Machine] Compiling circuit for {BACKEND}...")
    backend_info = find_backend(BACKEND)

    # 真机模式：限死比特池，避免插 SWAP
    if physical_qubits is not None:
        print(f"[Real Machine] Using physical qubits: {physical_qubits}")
        compiled_circuit = compile(circuit, backend_info=backend_info,
                                  available_qubits=physical_qubits)
    else:
        compiled_circuit = compile(circuit, backend_info=backend_info)

    # 2. 用原始线路计算理想态 psi（逻辑 qubit，n 个）
    from uniqc.simulator import Simulator
    sim_ideal = Simulator(backend_type='statevector')
    psi = sim_ideal.simulate_statevector(circuit)

    # 2.1 打印编译后的线路（前30行）
    print(f"[Real Machine] Compiled circuit originir (first 30 lines):")
    originir_lines = compiled_circuit.originir.strip().split('\n')[:30]
    for i, line in enumerate(originir_lines):
        print(f"  {i+1:2d}: {line}")

    # 2.2 统计门数（逐行匹配，避免 SXDG 误算成 SX）
    originir_full = compiled_circuit.originir
    from collections import Counter
    cnt = Counter()
    for line in originir_full.splitlines():
        line = line.strip()
        if not line or line.startswith(('QINIT', 'CREG', 'BARRIER', '//', 'MEASURE')):
            continue
        gate_name = line.split(None, 1)[0].upper()
        cnt[gate_name] += 1

    rphi_count = cnt.get('RPHI', 0)
    cz_count = cnt.get('CZ', 0)
    rz_count = cnt.get('RZ', 0)
    sx_count = cnt.get('SX', 0) + cnt.get('X1', 0)  # SX 和 X1 都算 RPhi 脉冲
    swap_count = cnt.get('SWAP', 0)
    g1_count = rphi_count + sx_count  # G1 = RPhi + SX + X1
    print(f"[Real Machine] Gate counts: RPhi={rphi_count}, SX+X1={sx_count}, CZ={cz_count}, RZ={rz_count}, SWAP={swap_count}")
    print(f"[Real Machine] G1={g1_count}, G2={cz_count}")

    # 验证 CZ 数量是否与预期相符（真机模式下应该有 0 个 SWAP）
    _cnt = {}
    for _i in range(M_CYCLES):
        _c = PATTERN_8CYCLE[_i % 8]
        _cnt[_c] = _cnt.get(_c, 0) + 1
    expected_cz = sum(len(color_assignment.get(_c, [])) * _cnt.get(_c, 0) for _c in color_assignment)
    if cz_count != expected_cz:
        raise RuntimeError(
            f"路由没消掉！实测 CZ={cz_count}，目标={expected_cz} —— "
            f"检查 available_qubits / find_physical_qubits 的结果"
        )

    # 解析测量映射 cbit -> 逻辑比特
    measure_map = {}  # c_idx -> q_idx (可能是物理编号)
    measure_lines = []
    for line in compiled_circuit.originir.split('\n'):
        line = line.strip()
        if line.startswith('MEASURE'):
            measure_lines.append(line)
            content = line[7:].strip()
            parts = content.split(',')
            if len(parts) >= 2:
                q_part = parts[0].strip()
                c_part = parts[1].strip()
                q_idx = int(q_part.split('[')[1].split(']')[0])
                c_idx = int(c_part.split('[')[1].split(']')[0])
                measure_map[c_idx] = q_idx

    # 打印原始 MEASURE 行
    print(f"[Real Machine] MEASURE lines (first 5):")
    for line in measure_lines[:5]:
        print(f"  {line}")

    # 归一化：将 q_idx 转为逻辑编号 (0~n-1)
    # q_idx 可能是物理编号 (39~86) 或逻辑编号 (0~9)
    n_qubits = n  # 使用传入的 n 参数
    normalized_map = {}
    for c_idx, q_idx in measure_map.items():
        if physical_qubits and q_idx in physical_qubits:
            # 物理编号 -> 逻辑编号
            logical_idx = physical_qubits.index(q_idx)
            normalized_map[c_idx] = logical_idx
        elif 0 <= q_idx < n_qubits:
            # 已经是逻辑编号
            normalized_map[c_idx] = q_idx
        else:
            raise ValueError(f"Invalid qubit {q_idx} (not in physical_qubits or 0~{n_qubits-1})")

    # 打印归一化后的 map（值域必须是 0~n-1）
    is_identity = all(normalized_map.get(i, i) == i for i in range(n_qubits))
    map_str = "(identity)" if is_identity else str(normalized_map)
    print(f"[Real Machine] Measure map (normalized): {map_str}")

    # 3. 提交到真机
    print(f"[Real Machine] Submitting circuit to {BACKEND} with {N_s} shots...")
    task_id = submit_task(
        compiled_circuit,
        backend=BACKEND,
        shots=N_s
    )
    print(f"[Real Machine] Task submitted, task_id = {task_id}")

    # 保存 task_id 到全局，供结果记录
    global LAST_TASK_ID
    LAST_TASK_ID = task_id

    # 4. 等待结果
    print(f"[Real Machine] Waiting for result...")
    result = wait_for_result(task_id)
    print(f"[Real Machine] Result received: {result}")

    # Step 1 验证模式：直接返回空结果
    # print("[Real Machine] Step 1: 验证模式，只编译不提交")

    # 4. 解析比特串
    # 解析比特串的工具函数（用 Python 标准库，不用外部依赖）
    # ============================================================
    # 【配置】真机 counts key 的比特序约定
    #   指纹测试结果：lsb_first (最右 = 逻辑比特0)
    # ============================================================
    REAL_KEY_ORDER = 'lsb_first'

    def parse_bitstring_key(key, n_bits):
        """把真机返回的 counts key 转成整数索引（bit0 = 逻辑比特 0）"""
        if isinstance(key, int):
            return key
        s = str(key).strip()
        # 二进制字符串
        raw = s[2:] if s[:2].lower() == '0b' else s
        if len(raw) == n_bits and set(raw) <= set('01'):
            # 归一化：lsb_first 时不需要反转
            cbits = raw[::-1] if REAL_KEY_ORDER == 'lsb_first' else raw
            out = 0
            for cbit, ch in enumerate(cbits):
                if ch == '1':
                    out |= (1 << cbit)
            return out
        # hex
        if s[:2].lower() == '0x':
            return int(s, 16)
        # 十进制
        return int(s)

    # result.counts 是 {bitstring: count} 字典
    n_qubits = circuit.qubit_num  # 逻辑比特数
    bitstrings = []
    if hasattr(result, 'counts') and result.counts:
        for bitstring, count in result.counts.items():
            # 步骤1: lsb_first 解析得到 cbit 序整数
            cidx = parse_bitstring_key(bitstring, n_qubits)
            # 步骤2: 按 normalized_map 置换为逻辑序整数
            logical_idx = 0
            for j in range(n_qubits):
                cbit_j = (cidx >> j) & 1
                logic_bit = normalized_map.get(j, j)
                if cbit_j:
                    logical_idx |= (1 << logic_bit)
            bitstrings.extend([logical_idx] * count)
    elif hasattr(result, 'probabilities'):
        # 如果只有概率，采样
        import numpy as np
        probs = np.array(list(result.probabilities.values()))
        keys = list(result.probabilities.keys())
        samples = np.random.choice(len(keys), size=N_s, p=probs)
        for s in samples:
            cidx = parse_bitstring_key(s, n_qubits)
            logical_idx = 0
            for j in range(n_qubits):
                cbit_j = (cidx >> j) & 1
                logic_bit = normalized_map.get(j, j)
                if cbit_j:
                    logical_idx |= (1 << logic_bit)
            bitstrings.append(logical_idx)

    logical_bitstrings = bitstrings

    print(f"[Real Machine] Got {len(bitstrings)} bitstrings from real machine")
    # 门数统计合并返回
    gate_counts = {'G1': g1_count, 'G2': cz_count}
    return logical_bitstrings, psi, gate_counts


# ============================================================
# 【模块 3：保真度计算】步骤 4 — 算 F_XEB
# ============================================================
# 函数功能：算 F_XEB
# ============================================================
# 计算公式：
#   p_expected(x) = |⟨x|ψ⟩|²（理论概率，从 |ψ⟩ 算）
#   p_measured(x) = (比特串 x 在 bitstrings 中出现次数) / N_s（实测频率，从 bitstrings 统计）
#   F_XEB = 2^n · Σ_i P_measured(x_i) · P_expected(x_i) - 1（加权和形式）
# ============================================================
# 参数说明：
#   psi        : 纯态矢量|ψ⟩（来自无噪声模拟）—— 用于算 p_expected
#   bitstrings : shape=(N_s,) 整数数组，比特串的整数编码 0 ~ 2ⁿ-1（来自采样）—— 用于算 p_measured
#   n          : qubit 数
# 返回值：
#   F_XEB : 交叉熵基准保真度
# ============================================================
def compute_F_XEB(psi, bitstrings, n, qubit_mapping=None):
    """计算 F_XEB

    Args:
        psi: 理想态矢量
        bitstrings: 测量比特串列表
        n: 量子比特数
        qubit_mapping: 可选，物理到逻辑 qubit 的映射 dict
    """
    # 1.根据 |ψ⟩ 算理论概率 p_expected(x) = |⟨x|ψ⟩|²（每个分量取模方）
    psi = np.array(psi)
    p_expected = np.abs(psi) ** 2

    # 如果有 qubit 映射，需要转换 bitstrings
    if qubit_mapping is not None:
        # qubit_mapping: {物理 qubit: 逻辑 qubit}
        # 将物理比特串转换为逻辑比特串
        mapped_bitstrings = []
        for bs in bitstrings:
            # bs 是整数，需要转成二进制
            binary = format(bs, f'0{n}b')
            # 按映射重新排列
            mapped = 0
            for phys, logical in qubit_mapping.items():
                if phys < n and logical < n:
                    if binary[phys] == '1':
                        mapped |= (1 << logical)
            mapped_bitstrings.append(mapped)
        bitstrings = mapped_bitstrings

    # 2.根据bitstrings 统计实测概率 p_measured(x) = (比特串 x 出现次数) / N_s
    unique, counts = np.unique(bitstrings, return_counts=True)
    p_measured = counts / len(bitstrings)

    # 3.只取前 2^n 个概率
    p_expected = p_expected[:2**n]

    # 4.加权和：Σ_i P_measured(x_i) · P_expected(x_i)
    # 需要对齐索引
    weight_sum = 0.0
    for bs, pm in zip(unique, p_measured):
        if bs < len(p_expected):
            weight_sum += pm * p_expected[bs]

    # 5.代入公式计算：F_XEB = 2^n · 加权和 - 1
    F_XEB = (2**n) * weight_sum - 1
    # 6.返回 F_XEB
    return F_XEB


# ============================================================
# 【模块 4：重复实验】步骤 5 — 重复 K 次取期望和标准差
# ============================================================
# 函数功能：重复 K 次实验，取 F_XEB 的期望和标准差
# ============================================================
# 参数说明：
#   n         : qubit 数
#   m         : cycle 数
#   use_noise : 是否使用噪声
#   N_s       : 采样次数
#   K         : 重复次数（读取全局变量 K_REPEATS）
# 返回值：
#   F_XEB_mean : F_XEB 的期望值
#   F_XEB_std : F_XEB 的标准差
# ============================================================


def _run_one_experiment(k, n, m, color_assignment, use_noise, N_s):
    """
    单次实验 worker：跑 1 个随机线路 + 1 次 run_circuit + 1 次 compute_F_XEB
    由 run_repeat_experiment 并行分发

    参数：
      k : int，随机种子（决定第 k 个随机线路）
      n : int，qubit 数
      m : int，cycle 数
      color_assignment : Dict，4色耦合对（来自 layout_gen）
      use_noise : bool，是否加噪声
      N_s : int，采样数

    返回：
      F_XEB : float，第 k 次实验的 F_XEB
    """
    random.seed(k)
    circuit = generate_random_circuit(n, m, color_assignment, seed=k)
    bitstrings, psi = run_circuit(circuit, use_noise, N_s, seed=k)
    F_XEB = compute_F_XEB(psi, bitstrings, n)
    return F_XEB


def run_repeat_experiment(n, m, color_assignment, use_noise, N_s, n_workers=None):
    """
    并行版：K 次实验用 ProcessPoolExecutor 并行分发
    K=10 + 32 worker → ~10x 加速
    """
    if n_workers is None:
        n_workers = min(cpu_count(), K_REPEATS)

    F_list = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_run_one_experiment, k, n, m, color_assignment, use_noise, N_s)
            for k in range(K_REPEATS)
        ]
        for future in as_completed(futures):
            try:
                F_XEB = future.result()
                F_list.append(F_XEB)
            except Exception as e:
                print(f"实验 {len(F_list)+1} 失败: {e}")
                raise

    F_XEB_mean = np.mean(F_list)
    F_XEB_std = np.std(F_list)
    return F_XEB_mean, F_XEB_std


# ============================================================
# 【模块 5：主函数】XEB 实验主流程
# ============================================================
# 函数功能：XEB 实验主流程
# ============================================================
# 参数说明：
#   n         : 量子比特数（若为 None，则使用全局变量 N_QUBITS）
#   use_noise : 是否加噪声（False=无噪声理想演化；True=含 Depolarizing 噪声演化）
#               噪声大小由顶部的 EPS1 / EPS2 控制
#   n_workers : 并行 worker 数
#   use_real_machine : 是否使用真机（默认 False，使用本地模拟器）
# 返回值：
#   F_mean : K 次 F_XEB 的均值
#   F_std  : K 次 F_XEB 的标准差
#   alpha_f : 理论预测的保真度（用于对比；无噪声时 = 1）
# ============================================================
def main(n=None, use_noise=False, n_workers=None, use_real_machine=False, dry_run=False):
    # 使用传入的 n 或默认的 N_QUBITS
    if n is None:
        n = N_QUBITS

    # 0.初始化错误率（先用默认值，后续真机模式会覆盖）
    eps1_err = EPS1_ERR
    eps2_err = EPS2_ERR
    eq_err = EQ_ERR
    layout_data = None
    chip_params_data = None

    # 1.加载布局（真机模式：读 JSON 或自动生成）
    if use_real_machine or dry_run:
        # 检查必要参数
        if LAYOUT_JSON_PATH is None and CHIP_PARAMS_JSON_PATH is None:
            raise ValueError("--layout and --chip-params are required for real machine mode")

        # 加载芯片参数
        if CHIP_PARAMS_JSON_PATH:
            chip_params_data = load_chip_params(CHIP_PARAMS_JSON_PATH)

        # 加载布局
        if LAYOUT_JSON_PATH:
            layout_data = load_layout_from_json(LAYOUT_JSON_PATH, n)
        elif CHIP_PARAMS_JSON_PATH:
            layout_data = select_qubits_from_json(CHIP_PARAMS_JSON_PATH, n)

        # 时效对账
        check_backend_alignment(layout_data, BACKEND)
        check_topology_alignment(layout_data, BACKEND)

        # 提取布局数据
        physical_qubits = layout_data['physical_qubits']
        color_assignment = layout_data['color_assignment']

        # 验证 color_assignment 的每一对都在 edges 中
        # edges 是逻辑编号，需要映射到物理边
        layout_edges = set()
        for e in layout_data['edges']:
            phys1 = physical_qubits[e[0]]
            phys2 = physical_qubits[e[1]]
            key = tuple(sorted([phys1, phys2]))
            layout_edges.add(key)
        for c in ['A', 'B', 'C', 'D']:
            for pair in color_assignment[c]:
                p1, p2 = pair
                phys1, phys2 = physical_qubits[p1], physical_qubits[p2]
                edge_key = tuple(sorted([phys1, phys2]))
                assert edge_key in layout_edges, \
                    f"Color {c} pair {pair} -> physical [{phys1},{phys2}] not in edges {layout_edges}"
        print(f"[Verify] color_assignment all pairs in edges: PASS")

        print(f"n={n}, physical_qubits={physical_qubits}")
        print(f"color_assignment={{")
        for c in ['A', 'B', 'C', 'D']:
            print(f"  '{c}': {color_assignment[c]},")
        print(f"}}")
    else:
        # 模拟模式：用原来的 layout_gen
        layout, color_assignment = generate_abcd_pairs(n)
        physical_qubits = None

    # 验证打印
    n_pairs = sum(len(color_assignment[c]) for c in ['A', 'B', 'C', 'D'])
    print(f"[Verify] n = {n}")
    print(f"[Verify] G_1 = n * M_CYCLES = {n} * {M_CYCLES} = {n * M_CYCLES}")
    print(f"[Verify] G_2 = sum(ABCD pairs) * M_CYCLES / 4 = {n_pairs} * {M_CYCLES} / 4 = {n_pairs * M_CYCLES / 4}")

    # 2.画 PNG 布局图
    all_couplings = color_assignment['A'] + color_assignment['B'] + color_assignment['C'] + color_assignment['D']
    if use_real_machine or dry_run:
        # 真机模式：把逻辑编号映射回物理编号
        phys_couplings = [(physical_qubits[a], physical_qubits[b]) for a, b in all_couplings]
        phys_colors = {
            c: [(physical_qubits[a], physical_qubits[b]) for a, b in color_assignment[c]]
            for c in ['A', 'B', 'C', 'D']
        }
        out_png = os.path.join(os.getcwd(), f"layout_n{n}_real.png")
        plot_layout([physical_qubits], phys_couplings, phys_colors, out_png)
    else:
        # 模拟模式：逻辑编号
        layout = compute_layout(n)
        out_png = os.path.join(os.getcwd(), f"layout_n{n}.png")
        plot_layout(layout, all_couplings, color_assignment, out_png)
    print(f"布局图已保存: {out_png}")

    # 3.打印实验模式（无噪声/含噪）+ 实验参数
    if use_noise:
        print(f"XEB Experiment - NOISY mode")
        print(f"Parameters: n={n}, m={M_CYCLES}, N_s={N_s}, K={K_REPEATS}")
        print(f"Injected p: eps1={EPS1}, eps2={EPS2}")
    else:
        print(f"XEB Experiment - IDEAL mode")
        print(f"Parameters: n={n}, m={M_CYCLES}, N_s={N_s}, K={K_REPEATS}")

    # 设计值门数
    G_1_design = n * M_CYCLES
    _cnt = {}
    for _i in range(M_CYCLES):
        _c = PATTERN_8CYCLE[_i % 8]
        _cnt[_c] = _cnt.get(_c, 0) + 1
    G_2_design = sum(len(color_assignment.get(_c, [])) * _cnt.get(_c, 0) for _c in color_assignment)

    # 4.alpha_f 计算（真机模式用 JSON 参数；2026-09-23 改1 新口径 + 改3 总时长）
    alpha_f_exact = 1.0
    eq_prod = gate1q_prod = cz_prod = deco_prod = 1.0
    if use_real_machine or dry_run:
        # 获取 backend_info
        backend_info = find_backend(BACKEND)

        # 编译样例电路: 提取电路总运行时长 T_总
        print("[Alpha F] Compiling sample circuit to measure T_total...")
        test_circuit = generate_random_circuit(n, M_CYCLES, color_assignment, seed=42)
        compiled_test = compile(test_circuit, backend_info=backend_info, available_qubits=physical_qubits)

        t_total_ns = schedule_total_duration_ns(compiled_test, backend_info)
        print(f"[Alpha F] T_total = {t_total_ns:.1f} ns")

        # 改2: 数据洞过滤
        check_data_holes(layout_data, chip_params_data)

        # 精确版（新口径: 保真度直接连乘 + 总时长退相干）
        result = compute_alpha_f_exact(
            layout_data, chip_params_data, n, M_CYCLES, color_assignment, t_total_ns)
        alpha_f_exact, eq_prod, gate1q_prod, cz_prod, deco_prod = result

        print(f"[Alpha F] alpha_f = {alpha_f_exact:.6f}")
        print(f"  ∏(1-e_q) = {eq_prod:.6f}")
        print(f"  ∏F1q^m = {gate1q_prod:.6f}")
        print(f"  ∏F_CZ^n_e = {cz_prod:.6f}")
        print(f"  ∏deco(q) = {deco_prod:.6f}")
    else:
        # 模拟器模式
        alpha_f_exact = 1.0

    # 5.DRY-RUN 模式：编译电路验证门数
    if dry_run:
        print(f"\n[DRY-RUN] Compiling circuit to verify gate counts...")
        # 生成 seed=0 的电路
        random.seed(0)
        circuit = generate_random_circuit(n, M_CYCLES, color_assignment, seed=0)

        # 编译
        backend_info = find_backend(BACKEND)
        compiled_circuit = compile(circuit, backend_info=backend_info,
                                  available_qubits=physical_qubits)

        # 统计门数
        from collections import Counter
        originir_full = compiled_circuit.originir
        cnt = Counter()
        for line in originir_full.splitlines():
            line = line.strip()
            if not line or line.startswith(('QINIT', 'CREG', 'BARRIER', '//', 'MEASURE')):
                continue
            gate_name = line.split(None, 1)[0].upper()
            cnt[gate_name] += 1

        rphi = cnt.get('RPHI', 0)
        sx = cnt.get('SX', 0) + cnt.get('X1', 0)
        cz = cnt.get('CZ', 0)
        rz = cnt.get('RZ', 0)
        swap = cnt.get('SWAP', 0)
        g1 = rphi + sx
        g2 = cz

        print(f"[DRY-RUN] Gate counts:")
        print(f"  RPhi={rphi}, SX+X1={sx}, CZ={cz}, RZ={rz}, SWAP={swap}")
        print(f"  G1={g1} (design={G_1_design}), G2={g2} (design={G_2_design})")

        # 解析测量映射
        measure_map = {}
        measure_lines = []
        for line in originir_full.split('\n'):
            line = line.strip()
            if line.startswith('MEASURE'):
                measure_lines.append(line)
                content = line[7:].strip()
                parts = content.split(',')
                if len(parts) >= 2:
                    q_part = parts[0].strip()
                    c_part = parts[1].strip()
                    q_idx = int(q_part.split('[')[1].split(']')[0])
                    c_idx = int(c_part.split('[')[1].split(']')[0])
                    measure_map[c_idx] = q_idx

        # 打印原始 MEASURE 行
        print(f"[DRY-RUN] MEASURE lines (first 5):")
        for line in measure_lines[:5]:
            print(f"  {line}")

        # 归一化
        normalized_map = {}
        for c_idx, q_idx in measure_map.items():
            if physical_qubits and q_idx in physical_qubits:
                normalized_map[c_idx] = physical_qubits.index(q_idx)
            elif 0 <= q_idx < n:
                normalized_map[c_idx] = q_idx
            else:
                raise ValueError(f"Invalid qubit {q_idx}")

        is_identity = all(normalized_map.get(i, i) == i for i in range(n))
        map_str = "(identity)" if is_identity else str(normalized_map)
        print(f"[DRY-RUN] Measure map (normalized): {map_str}")

        # 校验
        if swap > 0:
            raise RuntimeError(f"SWAP detected in dry-run! SWAP={swap}")
        if cz != G_2_design:
            raise RuntimeError(f"CZ count mismatch! got {cz}, expected {G_2_design}")

        print(f"\n[DRY-RUN] Validation complete, exiting without submitting tasks.")
        return None, None, alpha_f_exact

    # 6.真机模式：K 生效
    if use_real_machine:
        print(f"[Real Machine Mode] Using {BACKEND}, K={K_REPEATS}")

        results = []
        for k in range(K_REPEATS):
            print(f"\n--- Circuit {k+1}/{K_REPEATS} ---")
            # 生成电路
            random.seed(k)
            circuit = generate_random_circuit(n, M_CYCLES, color_assignment, seed=k)

            # 提交真机
            bitstrings, psi, gate_counts = run_circuit_on_real_machine(
                circuit, n, N_s, physical_qubits, color_assignment)

            G_1 = gate_counts['G1']
            G_2 = gate_counts['G2']

            # 门数明细打印
            print(f"[Gate Counts] G1={G_1} (design={G_1_design}), G2={G_2} (design={G_2_design})")
            if G_2 > G_2_design:
                raise RuntimeError(f"SWAP detected! G2={G_2} > design={G_2_design}")

            # F_XEB
            F_XEB = compute_F_XEB(psi, bitstrings, n)
            results.append({'seed': k, 'task_id': LAST_TASK_ID, 'G1': G_1, 'G2': G_2, 'F_XEB': F_XEB})
            print(f"F_XEB = {F_XEB:.6f}")

        # 汇总
        F_values = [r['F_XEB'] for r in results]
        F_mean = sum(F_values) / len(F_values)
        F_std = (sum((x - F_mean)**2 for x in F_values) / len(F_values)) ** 0.5 if len(F_values) > 1 else 0.0

        print(f"\n=== Summary ===")
        print(f"K={K_REPEATS}, F_mean={F_mean:.6f}, F_std={F_std:.6f}")
        print(f"alpha_f_exact={alpha_f_exact:.6f}")

        # 保存 JSON
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        result_json = {
            "backend": BACKEND,
            "n": n,
            "K": K_REPEATS,
            "results": results,
            "F_mean": F_mean,
            "F_std": F_std,
            "alpha_f_exact": alpha_f_exact,
        }
        result_path = f"xeb_results_{BACKEND.replace(':', '_')}_n{n}_K{K_REPEATS}_{timestamp}.json"
        with open(result_path, 'w') as f:
            json.dump(result_json, f, indent=2)
        print(f"Results saved to: {result_path}")

        return F_mean, F_std, alpha_f_exact

    # 7.模拟器模式
    F_mean, F_std = run_repeat_experiment(n, M_CYCLES, color_assignment, use_noise, N_s, n_workers=n_workers)
    alpha_f = 1.0 if not use_noise else alpha_f_exact

    print(f"Result: F_XEB = {F_mean:.6f} +/- {F_std:.6f}")
    print(f"Theory: alpha_f = {alpha_f:.6f}")

    return F_mean, F_std, alpha_f


# ============================================================
# 【命令行入口】使用案例
#   python xeb_v2_real.py            # 无噪声，默认 n=4（本地模拟器）
#   python xeb_v2_real.py --noise    # 含噪声
#   python xeb_v2_real.py --n 6     # n=6 量子比特
#   python xeb_v2_real.py --real     # 使用真机（originq:WK_C180）
#   python xeb_v2_real.py --real --n 4 --K 1  # 真机测试：n=4, K=1
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="单个量子比特数")
    parser.add_argument("--n_list", type=str, default=None,
                        help="多个 n 逗号分隔，如 10,12,14,16,18,20,22,24")
    parser.add_argument("--K", type=int, default=None,
                        help="重复次数（默认使用全局变量 K_REPEATS）")
    parser.add_argument("--noise", action="store_true", help="启用含噪声模拟")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="并行 worker 数（默认自动用 CPU 核数）")
    parser.add_argument("--real", action="store_true", help="使用真机运行（需要 ORIGINQ_API_KEY 环境变量）")
    parser.add_argument("--backend", type=str, default=None, help="真机后端，如 originq:WK_C180_2")
    parser.add_argument("--layout", type=str, default=None, help="布局 JSON 文件路径")
    parser.add_argument("--chip-params", type=str, default=None, help="芯片参数 JSON 文件路径")
    parser.add_argument("--ms", type=str, default=None,
                        help="深度列表逗号分隔, 如 5,10,15,20,25 (配合 --real, 自动整批提交)")
    parser.add_argument("--k", type=int, default=None, help="每档深度种子数 (默认 10)")
    parser.add_argument("--seed-base", type=int, default=70000,
                        help="种子段起点 (seed=seed_base+100*m+k)")
    parser.add_argument("--shots", type=int, default=2000, help="每任务 shots 数")
    parser.add_argument("--dry-run", action="store_true", help="走完全流程但不提交任务")
    parser.add_argument("--self-test", action="store_true", help="运行单元测试")
    args = parser.parse_args()

    # 自测模式
    if args.self_test:
        run_self_test()
        exit(0)

    # 设置全局后端
    BACKEND = args.backend if args.backend else BACKEND
    LAYOUT_JSON_PATH = args.layout
    CHIP_PARAMS_JSON_PATH = args.chip_params

    # 更新全局变量（临时覆盖）
    if args.K is not None:
        globals()['K_REPEATS'] = args.K

    # 深度扫描批量模式 (--ms): 整批提交, 结果按参数命名存档
    if args.ms:
        if not args.real:
            raise SystemExit('--ms 深度扫描需要 --real 真机模式')
        ld = load_layout_from_json(LAYOUT_JSON_PATH, args.n if args.n is not None else 0)
        ld['backend'] = BACKEND
        json.dump(ld, open(LAYOUT_JSON_PATH, 'w', encoding='utf-8'), indent=1)
        chip = load_chip_params(CHIP_PARAMS_JSON_PATH)
        check_backend_alignment(ld, BACKEND)
        check_topology_alignment(ld, BACKEND)
        check_data_holes(ld, chip)
        bi = find_backend(BACKEND)
        n_use = len(ld['physical_qubits']) if args.n is None else args.n
        ms_list = [int(x) for x in args.ms.split(',')]
        k_use = args.k if args.k else 10
        shots_use = args.shots if args.shots else 2000
        run_depth_scan(ld, chip, n_use, ms_list, k_use, args.seed_base, shots_use, bi,
                       dry_run=args.dry_run)
        raise SystemExit(0)

    import os
    import csv

    if args.n_list:
        # 批量模式：跑多个 n
        n_list = [int(x) for x in args.n_list.split(",")]
        results = []
        for n in n_list:
            print(f"\n{'='*60}")
            print(f"n = {n}")
            print('='*60)
            F_mean, F_std, alpha_f = main(n=n, use_noise=args.noise, n_workers=args.n_workers, use_real_machine=args.real, dry_run=args.dry_run)
            # 找 PNG 路径
            png_path = os.path.join(os.getcwd(), f"layout_n{n}.png")
            results.append((n, F_mean, F_std, alpha_f, png_path))

        # 写 CSV
        csv_path = os.path.join(os.getcwd(), "results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "F_XEB_mean", "F_XEB_std", "alpha_f", "deviation_pct", "layout_png"])
            for n, F_mean, F_std, alpha_f, png in results:
                dev = (F_mean - alpha_f) / alpha_f * 100
                writer.writerow([n, F_mean, F_std, alpha_f, f"{dev:.2f}", png])
        print(f"\n结果已保存到: {csv_path}")
    else:
        main(n=args.n, use_noise=args.noise, n_workers=args.n_workers, use_real_machine=args.real, dry_run=args.dry_run)

