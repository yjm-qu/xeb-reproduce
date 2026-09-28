# XEB 错误率标定实验 v2（复现 Arute 2019 流程）
# =====================================
# 目的：从比特串分布算 XEB fidelity，拟合得 ε₁/ε₂/e_q
#
# 4 步实验流程：
#   步骤 1   ：模块1标定单门错误率 ε₁（随机单门 + XEB）
#   步骤 2   ：模块2标定双门错误率 ε₂（随机单门+ISWAP + XEB）
#   步骤 3   ：模块3标定读出错误率 e_q（制备|0⟩/|1⟩ + 测量）
#   步骤 4   ：模块4用 Eq.77 预测保真度 F
#
# 3 步并行化改造：
#   改 1     ：模块1的 (m, seq) 任务并行 → ProcessPoolExecutor
#   改 2     ：模块2的 (layer, m, seq) 任务并行 → ProcessPoolExecutor
#   改 3     ：模块3、4 保持串行（计算量小）
#
# 命令行用法：
#   python calibrate_v2_pseudocode_2.0_parallel.py                          # 全参数运行
#   python calibrate_v2_pseudocode_2.0_parallel.py --n-qubits 5 --m-values 1,5  # 小参数测试
#   python calibrate_v2_pseudocode_2.0_parallel.py --output result.json   # 保存结果

# ============================================================================
# 实验参数（全局）
# ============================================================================
N_QUBITS = 20              # qubit 数量
M_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # m 序列
N_SEQUENCES = 10           # 每个 m 跑多少组随机序列
SHOTS = 1000               # 每组采样数
N_S = SHOTS                # 采样数（别名）

# 单门参数：sqrtX/sqrtY/sqrtW 通过 U3 实现
#   sqrtX = U3(π/2, -π/2, π/2)
#   sqrtY = U3(π/2, 0, 0)
#   sqrtW = U3(π/2, -π/4, π/4)
SINGLE_GATES = ['sqrtX', 'sqrtY', 'sqrtW']

# Layer 定义（来自 xeb_v2.py 的 ABCD_PAIRS，拓扑结构）
LAYER_A = [(1,12), (3,14), (4,5), (6,7), (8,9), (15,16), (17,18)]
LAYER_B = [(3,12), (5,14), (0,1), (7,8), (9,18), (10,11), (16,17)]
LAYER_C = [(7,16), (1,2), (3,4), (5,6), (11,12), (13,14), (18,19)]
LAYER_D = [(5,16), (7,18), (2,3), (1,10), (12,13), (14,15)]
LAYER_DEFS = {'A': LAYER_A, 'B': LAYER_B, 'C': LAYER_C, 'D': LAYER_D}

# 噪声参数（预设估值）
EPS1 = 0.16 / 100          # 单门 Pauli error（来自 Sycamore simultaneous 标定值）
EPS2 = 0.62 / 100          # 双门 Pauli error（来自 Sycamore simultaneous 标定值）

# ============================================================================
# 代码库调用
# ============================================================================
import numpy as np
import random
import argparse
import logging
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, NoisySimulator, ErrorLoader_GateTypeError, TwoQubitDepolarizing
from scipy.optimize import curve_fit


# ============================================================================
# 并行化辅助函数（模块级）
# ============================================================================
# 作用：每个 worker 进程独立执行的任务函数
# 原理：ProcessPoolExecutor 会把任务分发到不同进程，每个进程执行这个函数
# ============================================================================
# run_single_gate_experiment: 模块1的worker函数
# 参数说明：
#   args : tuple (m, seq, n_qubits, shots, eps1, seed)
#     - m         : int，序列长度（施加 m 个随机单门）
#     - seq       : int，第几组随机序列
#     - n_qubits  : int，qubit 数量
#     - shots     : int，每组采样数
#     - eps1      : float，单门错误率（Depolarizing p值）
#     - seed      : int，随机种子
# 返回值：
#   (m, f_xeb) : tuple，m值和对应的 F_XEB
# ============================================================================
def run_single_gate_experiment(args):
    """模块1的worker函数：跑单个(m, seq)实验"""
    m, seq, n_qubits, shots, eps1, seed = args

    # 固定随机种子保证可复现
    random.seed(seed + seq)

    # 构造电路
    circuit = Circuit(n_qubits)
    for q in range(n_qubits):
        for k in range(m):
            gate = random.choice(SINGLE_GATES)
            if gate == 'sqrtX':
                params = [np.pi/2, -np.pi/2, np.pi/2]
            elif gate == 'sqrtY':
                params = [np.pi/2, 0, 0]
            elif gate == 'sqrtW':
                params = [np.pi/2, -np.pi/4, np.pi/4]
            circuit.add_gate('U3', q, params=params)
    for q in range(n_qubits):
        circuit.measure(q)

    # 理想模拟
    ideal_sim = Simulator(backend_type='statevector')
    psi = ideal_sim.simulate_statevector(circuit)
    p_u = np.abs(np.array(psi)) ** 2

    # 噪声模拟
    error_model = ErrorLoader_GateTypeError(
        generic_error=[Depolarizing(p=0)],
        gatetype_error={'U3': [Depolarizing(p=eps1)]}
    )
    noisy_sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
    noisy_result = noisy_sim.simulate_shots(circuit.originir, shots=shots)

    # Per-qubit marginalize + F_XEB计算
    D = 2
    f_xeb_per_qubit = []
    for q in range(n_qubits):
        p_u_q = np.zeros(2)
        for b in range(len(p_u)):
            bit_q = (b >> q) & 1
            p_u_q[bit_q] += p_u[b]

        noisy_q = np.zeros(2)
        for b, count in noisy_result.items():
            bit_q = (b >> q) & 1
            noisy_q[bit_q] += count

        weight_sum_q = sum(p_u_q[bs] * noisy_q[bs] for bs in range(2)) / shots
        f_xeb_q = D * weight_sum_q - 1
        f_xeb_per_qubit.append(f_xeb_q)

    f_xeb = np.mean(f_xeb_per_qubit)
    print(f'[DIAG] m={m}, seq={seq}, f_xeb={f_xeb:.6f}', flush=True)
    return m, f_xeb


# ============================================================================
# run_two_gate_experiment: 模块2的worker函数
# 参数说明：
#   args : tuple (layer_name, m, seq, n_qubits, shots, eps1, eps2, seed)
#     - layer_name : str，layer 名称 ('A'/'B'/'C'/'D')
#     - m         : int，cycle 数
#     - seq       : int，第几组随机序列
#     - n_qubits  : int，qubit 数量
#     - shots     : int，每组采样数
#     - eps1      : float，单门错误率
#     - eps2      : float，双门错误率
#     - seed      : int，随机种子
# 返回值：
#   (layer_name, m, {(i,j): f_xeb}) : tuple，layer名、m值和该layer所有pair的F_XEB
# ============================================================================
def run_two_gate_experiment(args):
    """模块2的worker函数：跑单个(layer, m, seq)实验"""
    layer_name, m, seq, n_qubits, shots, eps1, eps2, seed = args

    # 固定随机种子
    random.seed(seed + seq)

    pair_list = LAYER_DEFS[layer_name]

    # 构造电路
    circuit = Circuit(n_qubits)
    for _ in range(m):
        # 单门层
        for q in range(n_qubits):
            gate = random.choice(SINGLE_GATES)
            if gate == 'sqrtX':
                params = [np.pi/2, -np.pi/2, np.pi/2]
            elif gate == 'sqrtY':
                params = [np.pi/2, 0, 0]
            elif gate == 'sqrtW':
                params = [np.pi/2, -np.pi/4, np.pi/4]
            circuit.add_gate('U3', q, params=params)
        # 双门层
        for (q1, q2) in pair_list:
            circuit.iswap(q1, q2)
    for q in range(n_qubits):
        circuit.measure(q)

    # 噪声模拟
    error_model = ErrorLoader_GateTypeError(
        generic_error=[Depolarizing(p=0)],
        gatetype_error={
            'U3': [Depolarizing(p=eps1)],
            'ISWAP': [TwoQubitDepolarizing(p=eps2)]
        }
    )
    noisy_sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
    noisy_result = noisy_sim.simulate_shots(circuit.originir, shots=shots)

    # 理想模拟
    ideal_sim = Simulator(backend_type='statevector')
    psi = ideal_sim.simulate_statevector(circuit)
    p_u = np.abs(np.array(psi)) ** 2

    # 对该layer所有pair计算F_XEB
    result = {}
    for (i, j) in pair_list:
        p_u_pair = {0: 0, 1: 0, 2: 0, 3: 0}
        for b in range(len(p_u)):
            bit_i = (b >> i) & 1
            bit_j = (b >> j) & 1
            pair_bs = (bit_i << 1) | bit_j
            p_u_pair[pair_bs] += p_u[b]

        noisy_pair = {0: 0, 1: 0, 2: 0, 3: 0}
        for b, count in noisy_result.items():
            bit_i = (b >> i) & 1
            bit_j = (b >> j) & 1
            pair_bs = (bit_i << 1) | bit_j
            noisy_pair[pair_bs] += count

        weight_sum = sum(p_u_pair[bs] * noisy_pair[bs] for bs in range(4)) / shots
        f_xeb = 4 * weight_sum - 1
        result[(i, j)] = f_xeb

    return layer_name, m, result


# ============================================================================
# 模块 1：标定单门错误率 ε₁（并行版）
# ============================================================================
# 函数功能：对 n_qubits 个 qubit 施加 m 个随机单门，测 XEB fidelity，拟合得 ε₁
# 并行方式：把 (m, seq) 组合拆成独立任务，用 ProcessPoolExecutor 并行执行
# 任务数 = len(m_values) × n_sequences
# ============================================================================
# 参数说明：
#   n_qubits    : int，qubit 数量（默认 20）
#   m_values    : list，m 序列（默认 [1,2,5,10,20,50,100,200,500,1000]）
#   n_sequences : int，每个 m 跑多少组随机序列（默认 10）
#   shots       : int，每组采样数（默认 2000）
#   n_workers   : int，并行 worker 数量（默认 cpu_count()）
#   eps1        : float，单门错误率（默认 0.0016）
#   logger      : logging.Logger，日志记录器
# 返回值：
#   eps1 : float，拟合得的 ε₁
# ============================================================================
def calibrate_single_gate_eps1_parallel(n_qubits=N_QUBITS, m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS, n_workers=None, eps1=EPS1, logger=None):
    if n_workers is None:
        n_workers = min(cpu_count(), len(m_values) * n_sequences)

    if logger:
        logger.info(f'模块1: {len(m_values)} m值 x {n_sequences} seq = {len(m_values) * n_sequences} 任务, {n_workers} workers')

    # 构建任务列表
    tasks = []
    for m in m_values:
        for seq in range(n_sequences):
            tasks.append((m, seq, n_qubits, shots, eps1, hash((m, seq)) % 100000))

    # 并行执行
    f_xeb_by_m = {m: [] for m in m_values}

    completed = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_gate_experiment, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                m, f_xeb = future.result()
                f_xeb_by_m[m].append(f_xeb)
                completed += 1
                if logger:
                    logger.info(f'[进度] 模块1 [{completed}/{total}] m={m}, seq={task[1]}, F_XEB={f_xeb:.6f}')
            except Exception as e:
                if logger:
                    logger.error(f'模块1任务失败: {e}')

    # 取平均
    f_xeb_avg_by_m = {m: np.mean(f_xeb_by_m[m]) for m in m_values}

    # 拟合
    m_array = np.array(list(f_xeb_avg_by_m.keys()))
    f_xeb_array = np.array(list(f_xeb_avg_by_m.values()))

    def f_xeb_model(m, eps):
        D = 2
        return (1 - eps / (1 - 1.0/D**2)) ** m

    popt, _ = curve_fit(f_xeb_model, m_array, f_xeb_array, p0=[0.01], bounds=(0, 1))
    return popt[0]


# ============================================================================
# 模块 2：标定双门错误率 ε₂（并行版）
# ============================================================================
# 函数功能：对 20 qubit 施加 m cycle 单门+ISWAP，marginalize 得 2-bit 分布，拟合得 ε₂
# 并行方式：把 (layer, m, seq) 组合拆成独立任务，用 ProcessPoolExecutor 并行执行
# 任务数 = len(LAYER_DEFS) × len(m_values) × n_sequences
# ============================================================================
# 参数说明：
#   n_qubits    : int，qubit 数量（默认 20）
#   m_values    : list，m 序列
#   n_sequences : int，每个 m 跑多少组随机序列
#   shots       : int，每组采样数
#   n_workers   : int，并行 worker 数量
#   eps1        : float，已标的单门错误率（用于拆出纯双门）
#   eps2        : float，双门错误率
#   logger      : logging.Logger，日志记录器
# 返回值：
#   (eps2_c, eps2) : tuple，ε₂ᶜ（总错误率）和 ε₂（纯双门错误率）
# ============================================================================
def calibrate_two_gate_eps2_parallel(n_qubits=N_QUBITS, m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS, n_workers=None, eps1=EPS1, eps2=EPS2, logger=None):
    if n_workers is None:
        n_workers = min(cpu_count(), len(m_values) * n_sequences * len(LAYER_DEFS))

    if logger:
        logger.info(f'模块2: {len(LAYER_DEFS)} layer x {len(m_values)} m x {n_sequences} seq = {len(LAYER_DEFS) * len(m_values) * n_sequences} 任务, {n_workers} workers')

    # 构建任务列表
    tasks = []
    for layer_name in LAYER_DEFS.keys():
        for m in m_values:
            for seq in range(n_sequences):
                tasks.append((layer_name, m, seq, n_qubits, shots, eps1, eps2, hash((layer_name, m, seq)) % 100000))

    # 并行执行
    pair_data = {}
    completed = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_two_gate_experiment, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                layer_name, m, f_xeb_dict = future.result()
                for (i, j), f_xeb in f_xeb_dict.items():
                    if (i, j) not in pair_data:
                        pair_data[(i, j)] = {m: [] for m in m_values}
                    pair_data[(i, j)][m].append(f_xeb)
                completed += 1
                avg_f_xeb = sum(f_xeb_dict.values()) / len(f_xeb_dict) if f_xeb_dict else 0
                if logger:
                    logger.info(f'[进度] 模块2 [{completed}/{total}] layer={layer_name}, m={m}, seq={task[2]}, F_XEB_avg={avg_f_xeb:.6f}')
            except Exception as e:
                if logger:
                    logger.error(f'模块2任务失败: {e}')

    # 拟合
    eps2_c_by_layer = {}
    for layer_name, pair_list in LAYER_DEFS.items():
        eps2_c_pair = {}
        for (i, j) in pair_list:
            f_xeb_curve = [np.mean(pair_data[(i, j)][m]) for m in m_values]
            m_array = np.array(m_values)
            f_array = np.array(f_xeb_curve)

            def f_xeb_model_pair(m, eps):
                D = 4
                return (1 - eps / (1 - 1/D**2)) ** m

            try:
                popt, _ = curve_fit(f_xeb_model_pair, m_array, f_array, p0=[0.01], bounds=(0, 1))
                eps2_c_pair[(i, j)] = popt[0]
            except RuntimeError:
                eps2_c_pair[(i, j)] = 0.01

        eps2_c_layer = np.mean(list(eps2_c_pair.values()))
        eps2_c_by_layer[layer_name] = eps2_c_layer

    eps2_c = np.mean(list(eps2_c_by_layer.values()))
    eps2 = eps2_c - 2 * eps1

    return eps2_c, eps2


# ============================================================================
# 模块 3：标定读出错误率 e_q（串行版）
# ============================================================================
# 函数功能：制备已知态 |0⟩/|1⟩，测量比对得读出错误率
# 说明：计算量极小（仅2次测量），无需并行
# ============================================================================
# 参数说明：
#   n_qubits : int，qubit 数量（默认 1）
#   shots    : int，每种态采样数（默认 3000）
# 返回值：
#   (e_0, e_1, e_q_mean) : tuple
#     - e_0      : float，制备 |0⟩ 测成 |1⟩ 的错误率
#     - e_1      : float，制备 |1⟩ 测成 |0⟩ 的错误率
#     - e_q_mean : float，平均读出错误率 (e_0 + e_1) / 2
# ============================================================================
def calibrate_readout_eq(n_qubits=1, shots=3000):
    results = {}
    for state in [0, 1]:
        circuit = Circuit(n_qubits)
        if state == 1:
            circuit.x(0)
        circuit.measure(0)

        sim = Simulator(backend_type='statevector')
        result = sim.simulate_shots(circuit.originir, shots=shots)

        correct_count = result.get(state, 0)
        error_count = shots - correct_count
        results[state] = error_count / shots

    e_0 = results[0]
    e_1 = results[1]
    e_q_mean = (e_0 + e_1) / 2
    return e_0, e_1, e_q_mean


# ============================================================================
# 模块 4：Eq.77 预测 fidelity（串行版）
# ============================================================================
# 函数功能：代入 ε₁/ε₂/e_q 预测 F，与实际跑霸权电路对比
# 说明：仅几次乘法运算，无需并行
# ============================================================================
# 参数说明：
#   eps1           : float，单门错误率
#   eps2           : float，双门错误率
#   eq             : float，读出错误率
#   n_single_gates : int，电路中单门数
#   n_two_gates    : int，电路中双门数
#   n_measurements : int，测量数
# 返回值：
#   F : float，预测的 fidelity
# ============================================================================
def predict_fidelity_eq77(eps1, eps2, eq, n_single_gates, n_two_gates, n_measurements):
    F = (1 - eps1) ** n_single_gates
    F *= (1 - eps2) ** n_two_gates
    F *= (1 - eq) ** n_measurements
    return F


# ============================================================================
# 主函数
# ============================================================================
# 函数功能：依次运行模块1→2→3→4，输出 ε₁, ε₂, e_q, F
# 并行调度：模块1/2 用 ProcessPoolExecutor，模块3/4 串行
# 命令行参数：通过 argparse 传入，支持小参数测试和全参数运行
# ============================================================================
# 参数说明（通过 argparse 传入）：
#   --n-qubits     : int，qubit 数量（默认 20）
#   --m-values    : str，m 序列，逗号分隔（默认 "1,2,5,10,20,50,100,200,500,1000"）
#   --n-sequences : int，每个 m 跑多少组（默认 10）
#   --shots       : int，每组采样数（默认 2000）
#   --n-workers   : int，并行 worker 数量（默认 cpu_count()）
#   --eps1        : float，单门错误率（默认 0.0016）
#   --eps2        : float，双门错误率（默认 0.0062）
#   --output      : str，结果输出文件路径（默认 None，不保存）
# 返回值：
#   (eps1, eps2_c, eps2, e_q, F) : tuple
# ============================================================================
def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='XEB 错误率标定实验')
    parser.add_argument('--n-qubits', type=int, default=N_QUBITS)
    parser.add_argument('--m-values', type=str, default=','.join(map(str, M_VALUES)))
    parser.add_argument('--n-sequences', type=int, default=N_SEQUENCES)
    parser.add_argument('--shots', type=int, default=SHOTS)
    parser.add_argument('--n-workers', type=int, default=None)
    parser.add_argument('--eps1', type=float, default=EPS1)
    parser.add_argument('--eps2', type=float, default=EPS2)
    parser.add_argument('--output', type=str, default=None, help='结果输出文件')
    args = parser.parse_args()

    # 解析 m_values
    m_values = [int(x) for x in args.m_values.split(',')]

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info(f'参数: n_qubits={args.n_qubits}, m_values={m_values}, n_sequences={args.n_sequences}, shots={args.shots}')
    logger.info(f'n_workers: {args.n_workers or cpu_count()}')

    # 模块1
    logger.info('[模块1] 开始...')
    eps1 = calibrate_single_gate_eps1_parallel(
        n_qubits=args.n_qubits,
        m_values=m_values,
        n_sequences=args.n_sequences,
        shots=args.shots,
        n_workers=args.n_workers,
        eps1=args.eps1,
        logger=logger
    )
    logger.info(f'[模块1] ε₁ = {eps1:.6f}')

    # 模块2
    logger.info('[模块2] 开始...')
    eps2_c, eps2 = calibrate_two_gate_eps2_parallel(
        n_qubits=args.n_qubits,
        m_values=m_values,
        n_sequences=args.n_sequences,
        shots=args.shots,
        n_workers=args.n_workers,
        eps1=eps1,
        eps2=args.eps2,
        logger=logger
    )
    logger.info(f'[模块2] ε₂ᶜ = {eps2_c:.6f}, ε₂ = {eps2:.6f}')

    # 模块3
    logger.info('[模块3] 开始...')
    e_0, e_1, e_q = calibrate_readout_eq(n_qubits=1, shots=3000)
    logger.info(f'[模块3] e_0 = {e_0:.6f}, e_1 = {e_1:.6f}, e_q = {e_q:.6f}')

    # 模块4：计算门数并用 Eq.77 预测保真度
    #   G_1 = n*m = 20*20 = 400（单量子门总数）
    G_1 = 400
    #   G_2 = 27*m/4 = 27*20/4 = 135（双量子门总数，ABCD 平均）
    G_2 = 135
    #   n_measurements = n = 20（测量数，每个 qubit 测 1 次）
    n_measurements = 20
    #   F = (1-ε₁)^G₁ · (1-ε₂)^G₂ · (1-e_q)^n_measurements
    F = predict_fidelity_eq77(eps1, eps2, e_q, G_1, G_2, n_measurements)
    logger.info(f'[模块4] F = {F:.6e}')

    # 汇总
    logger.info(f'\n===== 汇总 =====')
    logger.info(f'ε₁ = {eps1:.6f}, ε₂ᶜ = {eps2_c:.6f}, ε₂ = {eps2:.6f}, e_q = {e_q:.6f}, F = {F:.6e}')

    # 保存结果
    if args.output:
        result = {
            'eps1': eps1,
            'eps2_c': eps2_c,
            'eps2': eps2,
            'e_q': e_q,
            'F': F
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f'结果已保存到 {args.output}')

    return eps1, eps2_c, eps2, e_q, F


# ============================================================================
# 命令行入口
# ============================================================================
# 用法：
#   python calibrate_v2_pseudocode_2.0_parallel.py                          # 全参数
#   python calibrate_v2_pseudocode_2.0_parallel.py --n-qubits 5       # 小参数
#   python calibrate_v2_pseudocode_2.0_parallel.py --output result.json # 保存结果
# ============================================================================
if __name__ == '__main__':
    main()
