# -*- coding: utf-8 -*-
"""
XEB 真机实验主程序（瘦身版，2026-10-08）
============================================================
目的：在 WK_C180 真机上按深度扫描跑随机线路采样（RCS）实验，
     逐线路算 F_XEB，与逐比特/逐边 α 预测对比。
流程（四步）：
  ① 构建线路：n 比特 × m cycle；每 cycle = 每 qubit 一个随机单门
     (√X/√Y/√W 相邻不重) + 按 ABCACB 3色轮转拍一组 CZ
  ② 提交真机：--ms 各深度 × --k 条种子线路整批提交
  ③ 算保真度：本地理想态矢 psi + 实测 counts(经 cbit->逻辑映射) → F_XEB
  ④ 出对比表：F 均值±SEM vs α(读出×单门×双门，逐元素，无④退相干)
用法：
  # 小规模管道测试（假实测表）
  python3 xeb_v2_real.py --real --backend originq:WK_C180 \
      --layout selected_layout_n13A_3col.json \
      --chip-params measured_params.json \
      --ms 21 --k 3 --shots 2000 --seed-base 80000
  # 加 --dry-run 只本地自检不提交；--self-test 跑单元测试
瘦身说明：原 1277 行裁掉模拟器机制/旧单实验路径/死常量（原件在
  archive/xeb_v2_real_full.py），回归锚点 anchors_before/after.json 比对通过。
"""
import argparse
import json
import os
import random
import time

import numpy as np
from datetime import datetime, timezone

from uniqc import Circuit, compile, submit_batch, wait_for_result
from uniqc.backend_adapter import find_backend
from uniqc.simulator import Simulator

from tests import run_self_test
from layout_gen import select_qubits_from_json

# ============================================================
# 真机配置（API key：环境变量优先，其次 uniqc 配置系统）
# ============================================================
ORIGINQ_API_KEY = os.environ.get("ORIGINQ_API_KEY", "")
if not ORIGINQ_API_KEY:
    try:
        from uniqc.config import load_originq_config
        _cfg = load_originq_config()
        ORIGINQ_API_KEY = _cfg.get("api_key", "") or _cfg.get("token", "")
    except Exception:
        ORIGINQ_API_KEY = ""

BACKEND = "originq:WK_C180_2"                      # 默认后端（跑 WK_C180 时 --backend 覆盖）
LAYOUT_JSON_PATH = "selected_layout_n13A_3col.json"  # 现役布局（A 版 3 色）
CHIP_PARAMS_JSON_PATH = "../chip_params_WK_C180.json"  # 官方参数正本（父目录）
LAST_TASK_ID = None                                 # 最近一次提交的 task_id

print("[Real Machine Config]")
print("  API Key loaded:", bool(ORIGINQ_API_KEY))
print("  Default Backend:", BACKEND)

# ============================================================
# 模块一：加载与预检（布局、芯片参数、后端对齐、拓扑时效、数据洞）
# ============================================================
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



# ============================================================
# 模块二：深度扫描主流程（构建->提交->打分->汇总，逐线路自检）
# ============================================================
def run_depth_scan(layout_data, chip_params, n, ms, k, seed_base, shots,
                    backend, backend_info, dry_run=False):
    """深度扫描实验（2026-09-23 CLI 收编版）: ms x k 条种子线路, 整批提交.
    逐线路自检: CZ 计数与染色一致 / 零 SWAP / 测量映射.
    返回汇总 dict (dry_run 模式返回 None, 只打印自检与 alpha)."""
    phys = layout_data['physical_qubits']
    ca = layout_data['color_assignment']
    jobs = []
    for m in ms:
        cc = {}
        for i in range(m):
            cc[PATTERN_CYCLE[i % 6]] = cc.get(PATTERN_CYCLE[i % 6], 0) + 1
        expected_cz = sum(len(ca[c]) * cc.get(c, 0) for c in ca)
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

    # 关两级自动编译，与 uniqc 版标定脚本同配置（2026-10-08 统一口径）：
    # local_compile=0 跳过本地二次优化；cloud_compile=0 请求云端不自动编译
    task_id = submit_batch([j['compiled'] for j in jobs], backend=backend,
                           shots=shots, local_compile=0, cloud_compile=0)
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

    print('===== 汇总: n=%d 深度扫描 (α = ①读出×②单门×③双门, 无④) =====' % n, flush=True)
    print('| m | F 均值±SEM | α | 比值 | σ偏离 |', flush=True)
    print('|---|---|---|---|---|', flush=True)
    summary = []
    for m in ms:
        fs = np.array([r['F'] for r in rows if r['m'] == m])
        al = np.array([r['alpha'] for r in rows if r['m'] == m])
        fm = fs.mean()
        se = fs.std(ddof=1) / np.sqrt(len(fs))
        a4 = al.mean()
        ratio = ('%.2f' % (fm / a4)) if (a4 > 1e-4 and fm > 0) else '—'
        s4 = (fm - a4) / se
        summary.append({'m': m, 'K': len(fs), 'F_mean': round(fm, 4), 'F_SEM': round(se, 4),
                        'alpha_mean': round(a4, 4)})
        print('| %d | %.4f±%.4f | %.4f | %s | %+.1fσ |' %
              (m, fm, se, a4, ratio, s4), flush=True)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_file = 'depthscan_n%d_k%d_%s.json' % (n, k, stamp)
    json.dump({'backend': BACKEND, 'n': n, 'K': k, 'ms': ms, 'seed_base': seed_base,
               'shots': shots, 'cluster': phys, 'rows': rows, 'summary': summary},
              open(out_file, 'w'), indent=1)
    print('saved', out_file, flush=True)
    print('ALL DONE', flush=True)
    return None



# ============================================================
# 模块三：α 预测（逐比特读出 × 逐比特单门 × 逐边双门；无④退相干）
# ============================================================
def compute_alpha_f_exact(layout_data, chip_params, n, m, color_assignment, t_total_ns=None):
    """
    精确版 alpha_f 计算（2026-09-23 改⑥: 去除④退相干项，老师拍板）
    alpha_f = ∏(1−e_q) × ∏F1q_q^m × ∏F_CZ_e^{n_e}

    - 门保真度按"存活率"直接连乘, 无 2x/4/3 换算（口径 B, 用户拍板）
    - ④退相干项已删除（此前为调度表总时长版）
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
    pattern = PATTERN_CYCLE  # 3色版；D 类恒空，color_counts[3] 恒 0
    color_counts = [0, 0, 0, 0]
    for i in range(m):
        color_idx = ['A', 'B', 'C', 'D'].index(pattern[i % 6])
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

    # ④ 退相干项已去除（老师 2026-09-23 拍板）：预测公式 = ①读出 × ②单门 × ③双门
    deco_product = 1.0
    alpha_f_exact = eq_product * gate1q_product * cz_product
    return alpha_f_exact, eq_product, gate1q_product, cz_product, deco_product




# ============================================================
# 模块四：门定义（√X/√Y/√W = RPHI(π/2, φ)；3色轮转 ABCACB）
# ============================================================
#   √X = 'sx' / 'sqrtX' / 'X90'
#   √Y = 'sy' / 'sqrtY' / 'Y90'
#   √W = 'sw' / 'sqrtW' / 'XY90'（其中 W=(X+Y)/√2）
# 填错时 simulate 不报错但结果不对——跑无噪声版本 F≈1 即可确认。
SINGLE_GATES = ['sqrtX', 'sqrtY', 'sqrtW']  # √X, √Y, √W，其中 W=(X+Y)/√2

PATTERN_CYCLE = ['A', 'B', 'C', 'A', 'C', 'B']  # 3色轮转序列（ABCACB，无同色连拍；2026-10-08 用户拍板，原 ABCBCA/4色ABCDCDAB 均作废）



# ============================================================
# 模块五：随机线路生成（每 cycle 单门层 + 双门层）
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
    #   4.2 双量子门层：按 ABCACB 序列查本 cycle 应使用的子集，遍历子集内所有 qubit 对施加双门
        #   4.2.1 确定本序列字母：将cycle序号对6取模，查 ABCACB 序列得本 cycle 子集对应的字母（A/B/C）
        letter = PATTERN_CYCLE[cycle % 6]
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
# 模块六：F_XEB 计算（线性式 2^n·Σ P_meas·P_exp − 1）
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
# 模块七：命令行入口（--real --ms 深度扫描 / --self-test）
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='XEB 真机深度扫描（瘦身版）')
    parser.add_argument("--real", action="store_true", help="真机模式（本程序仅支持真机）")
    parser.add_argument("--backend", type=str, default=None, help="如 originq:WK_C180")
    parser.add_argument("--layout", type=str, default=None, help="布局 JSON 路径")
    parser.add_argument("--chip-params", type=str, default=None, help="芯片参数 JSON（默认父目录官方正本；实测表由此传入）")
    parser.add_argument("--n", type=int, default=None, help="比特数（默认=布局自带）")
    parser.add_argument("--ms", type=str, default=None, help="深度列表逗号分隔, 如 21")
    parser.add_argument("--k", type=int, default=10, help="每档深度种子数")
    parser.add_argument("--seed-base", type=int, default=70000, help="种子段起点 (seed=seed_base+100*m+k)")
    parser.add_argument("--shots", type=int, default=2000, help="每任务 shots 数")
    parser.add_argument("--dry-run", action="store_true", help="走完全流程但不提交")
    parser.add_argument("--self-test", action="store_true", help="运行单元测试")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        raise SystemExit(0)
    if not args.ms:
        raise SystemExit('本程序为真机深度扫描专用：需同时给 --real 与 --ms（如 --ms 21）；'
                         '模拟器实验请用 archive/ 下的 xeb_v2.py')

    backend = args.backend if args.backend else BACKEND
    layout_path = args.layout if args.layout else LAYOUT_JSON_PATH
    chip_path = args.chip_params if args.chip_params else CHIP_PARAMS_JSON_PATH

    ld = load_layout_from_json(layout_path, args.n if args.n is not None else 0)
    ld['backend'] = backend
    json.dump(ld, open(layout_path, 'w', encoding='utf-8'), indent=1)
    chip = load_chip_params(chip_path)
    check_backend_alignment(ld, backend)
    check_topology_alignment(ld, backend)
    check_data_holes(ld, chip)
    bi = find_backend(backend)
    n_use = len(ld['physical_qubits']) if args.n is None else args.n
    ms_list = [int(x) for x in args.ms.split(',')]
    run_depth_scan(ld, chip, n_use, ms_list, args.k, args.seed_base, args.shots,
                   backend, bi, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
