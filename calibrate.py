# XEB 错误率标定实验 v2（复现 Arute 2019 流程）
# =====================================
# 目的：从比特串分布算 XEB fidelity，拟合得 ε₁/ε₂/e_q
#
# 修改点（isolated_neill 版本）：
#   改 1     ：电路改为 isolated 模式（单门 Circuit(1)，双门 Circuit(2)）
#   改 2     ：F_XEB 公式改为 Neill 2017 相对熵形式
#
# 命令行用法：
#   python calibrate_v2_pseudocode_2.0_isolated_neill.py                          # 全参数运行
#   python calibrate_v2_pseudocode_2.0_isolated_neill.py --m-values 1,5  # 小参数测试
#   python calibrate_v2_pseudocode_2.0_isolated_neill.py --output result.json   # 保存结果

# ============================================================================
# 实验参数（全局）
# ============================================================================
N_QUBITS = 20              # qubit 数量（仅用于模块3/4的路径相关参数）
M_VALUES = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1000]  # m 序列（去掉 1，避开 m=1 时 F_XEB 公式在均匀分布下的退化；小 m 区域对数间隔；大 m 区域加 750 让拟合尾部更准）
N_SEQUENCES = 10           # 每个 m 跑多少组随机序列
SHOTS = 2000               # 每组采样数
N_S = SHOTS                # 采样数（别名）

# 单门参数：sqrtX/sqrtY/sqrtW 通过 U3 实现
#   sqrtX = U3(π/2, -π/2, π/2)
#   sqrtY = U3(π/2, 0, 0)
#   sqrtW = U3(π/2, -π/4, π/4)
SINGLE_GATES = ['sqrtX', 'sqrtY', 'sqrtW']

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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, NoisySimulator, ErrorLoader_GateTypeError, TwoQubitDepolarizing
from scipy.optimize import curve_fit


# ============================================================================
# Neill 2017 F_XEB 计算函数
# ============================================================================
# S(P) = 自熵 = -Σ p_i · log₂ p_i（单参数）
# S(P, Q) = 交叉熵 = -Σ p_i · log₂ q_i（双参数，第一个决定 bin，第二个进 log）
# F_XEB = (S(P_incoherent, P_expected) - S(P_measured, P_expected)) / (S(P_incoherent, P_expected) - S(P_expected))
# ============================================================================
def self_entropy(p):
    """自熵 S(P) = -Σ p_i · log₂ p_i"""
    p = np.array(p, dtype=np.float64)
    # 用 mask 过滤 p=0 的项
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))


def cross_entropy(p, q):
    """交叉熵 S(P, Q) = -Σ p_i · log₂ q_i"""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    # 用 mask 过滤 p=0 或 q=0 的项
    mask = (p > 0) & (q > 0)
    return -np.sum(p[mask] * np.log2(q[mask]))


def compute_fxeb_neill(p_measured, p_expected, n):
    """
    用 Neill 2017 相对熵形式计算 F_XEB

    参数：
      p_measured : list[float]，实测概率分布（count / N_s）
      p_expected : list[float]，理想概率分布（statevector 算出来）
      n          : int，qubit 数（单门 n=1，双门 n=2）

    返回：
      F_XEB : float
    """
    D = 2 ** n  # 希尔伯特空间维度

    # P_expected: 理想分布
    p_exp = np.array(p_expected, dtype=np.float64)

    # P_measured: 实测分布（频率）
    p_meas = np.array(p_measured, dtype=np.float64)

    # P_incoherent: 均匀分布
    p_incoh = np.full(D, 1.0 / D, dtype=np.float64)

    # 计算三个熵
    S_exp = self_entropy(p_exp)              # S(P_expected)，自熵
    S_meas_exp = cross_entropy(p_meas, p_exp)  # S(P_measured, P_expected)，交叉熵
    S_incoh_exp = cross_entropy(p_incoh, p_exp)  # S(P_incoherent, P_expected)，交叉熵

    # F_XEB = (S(P_incoh, P_exp) - S(P_meas, P_exp)) / (S(P_incoh, P_exp) - S(P_exp))
    numerator = S_incoh_exp - S_meas_exp
    denominator = S_incoh_exp - S_exp

    if denominator == 0:
        # 避免除零
        return 0.0

    f_xeb = numerator / denominator
    return f_xeb


# ============================================================================
# 并行化辅助函数（模块级）
# ============================================================================
# run_single_gate_experiment: 模块1的worker函数（isolated 模式）
# 参数说明：
#   args : tuple (m, seq, shots, eps1, seed)
#     - m         : int，序列长度（施加 m 个随机单门）
#     - seq       : int，第几组随机序列
#     - shots     : int，每组采样数
#     - eps1      : float，单门错误率（Depolarizing p值）
#     - seed      : int，随机种子
# 返回值：
#   (m, f_xeb) : tuple，m值和对应的 F_XEB
# ============================================================================
def run_single_gate_experiment(args):
    """模块1的worker函数：跑单个(m, seq)实验（isolated 模式，1 qubit）"""
    m, seq, shots, eps1, seed = args

    # 固定随机种子保证可复现
    random.seed(seed + seq)

    # 构造电路：Circuit(1)，只对 1 个 qubit 加门
    # 约束：相邻 cycle 不能使用同一个单量子门
    circuit = Circuit(1)  # 关键改动：Circuit(1) 而非 Circuit(n_qubits)
    prev_gate = None
    for k in range(m):
        # 随机选门，但不能与上一个 cycle 相同
        available_gates = [g for g in SINGLE_GATES if g != prev_gate]
        gate = random.choice(available_gates)
        prev_gate = gate
        if gate == 'sqrtX':
            params = [np.pi/2, -np.pi/2, np.pi/2]
        elif gate == 'sqrtY':
            params = [np.pi/2, 0, 0]
        elif gate == 'sqrtW':
            params = [np.pi/2, -np.pi/4, np.pi/4]
        circuit.add_gate('U3', 0, params=params)  # 只有一个 qubit，索引为 0
    circuit.measure(0)

    # 理想模拟
    ideal_sim = Simulator(backend_type='statevector')
    psi = ideal_sim.simulate_statevector(circuit)
    p_u = np.abs(np.array(psi)) ** 2  # 2^1 = 2 维

    # 噪声模拟
    error_model = ErrorLoader_GateTypeError(
        generic_error=[Depolarizing(p=0)],
        gatetype_error={'U3': [Depolarizing(p=eps1)]}
    )
    noisy_sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
    noisy_result = noisy_sim.simulate_shots(circuit.originir, shots=shots)

    # 把 noisy_result 转换为概率分布
    p_meas = np.zeros(2)
    for b, count in noisy_result.items():
        p_meas[b] = count / shots

    # 用 Neill 2017 相对熵形式计算 F_XEB
    n = 1  # isolated 模式：1 qubit
    f_xeb = compute_fxeb_neill(p_meas, p_u, n)

    print(f'[DIAG] m={m}, seq={seq}, f_xeb={f_xeb:.6f}', flush=True)
    return m, f_xeb


# ============================================================================
# run_two_gate_experiment: 模块2的worker函数（isolated 模式）
# 参数说明：
#   args : tuple (m, seq, shots, eps1, eps2, seed)
#     - m         : int，cycle 数
#     - seq       : int，第几组随机序列
#     - shots     : int，每组采样数
#     - eps1      : float，单门错误率
#     - eps2      : float，双门错误率
#     - seed      : int，随机种子
# 返回值：
#   (m, f_xeb) : tuple，m值和对应的 F_XEB
# ============================================================================
def run_two_gate_experiment(args):
    """模块2的worker函数：跑单个(m, seq)实验（isolated 模式，2 qubit）"""
    m, seq, shots, eps1, eps2, seed = args

    # 固定随机种子
    random.seed(seed + seq)

    # 构造电路：Circuit(2)，只对 2 个 qubit 加门
    # 约束：每个 qubit 相邻 cycle 不能使用同一个单量子门
    # 每个 cycle：1 个 U3 (q0) + 1 个 U3 (q1) + 1 个 ISWAP
    circuit = Circuit(2)  # 关键改动：Circuit(2) 而非 Circuit(n_qubits)
    prev_gate0 = None  # q0 上一个 cycle 用的门
    prev_gate1 = None  # q1 上一个 cycle 用的门
    for _ in range(m):
        # q0 上的随机 U3（不能与上一个 cycle 相同）
        available_gates0 = [g for g in SINGLE_GATES if g != prev_gate0]
        gate0 = random.choice(available_gates0)
        prev_gate0 = gate0
        if gate0 == 'sqrtX':
            params0 = [np.pi/2, -np.pi/2, np.pi/2]
        elif gate0 == 'sqrtY':
            params0 = [np.pi/2, 0, 0]
        elif gate0 == 'sqrtW':
            params0 = [np.pi/2, -np.pi/4, np.pi/4]
        circuit.add_gate('U3', 0, params=params0)

        # q1 上的随机 U3（不能与上一个 cycle 相同）
        available_gates1 = [g for g in SINGLE_GATES if g != prev_gate1]
        gate1 = random.choice(available_gates1)
        prev_gate1 = gate1
        if gate1 == 'sqrtX':
            params1 = [np.pi/2, -np.pi/2, np.pi/2]
        elif gate1 == 'sqrtY':
            params1 = [np.pi/2, 0, 0]
        elif gate1 == 'sqrtW':
            params1 = [np.pi/2, -np.pi/4, np.pi/4]
        circuit.add_gate('U3', 1, params=params1)

        # ISWAP
        circuit.iswap(0, 1)

    circuit.measure(0)
    circuit.measure(1)

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
    p_u = np.abs(np.array(psi)) ** 2  # 2^2 = 4 维

    # 把 noisy_result 转换为概率分布
    p_meas = np.zeros(4)
    for b, count in noisy_result.items():
        p_meas[b] = count / shots

    # 用 Neill 2017 相对熵形式计算 F_XEB
    n = 2  # isolated 模式：2 qubit
    f_xeb = compute_fxeb_neill(p_meas, p_u, n)

    return m, f_xeb


# ============================================================================
# 模块 1：标定单门错误率 ε₁（并行版，isolated 模式）
# ============================================================================
# 函数功能：对 1 个 qubit 施加 m 个随机单门，测 XEB fidelity，拟合得 ε₁
# 并行方式：把 (m, seq) 组合拆成独立任务，用 ProcessPoolExecutor 并行执行
# 任务数 = len(m_values) × n_sequences
# ============================================================================
# 参数说明：
#   m_values    : list，m 序列（默认 [1,2,5,10,20,50,100,200,500,1000]）
#   n_sequences : int，每个 m 跑多少组随机序列（默认 10）
#   shots       : int，每组采样数（默认 2000）
#   n_workers   : int，并行 worker 数量（默认 cpu_count()）
#   eps1        : float，单门错误率（默认 0.0016）
#   logger      : logging.Logger，日志记录器
# 返回值：
#   eps1 : float，拟合得的 ε₁
# ============================================================================
def calibrate_single_gate_eps1_parallel(m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS, n_workers=None, eps1=EPS1, logger=None):
    if n_workers is None:
        n_workers = min(cpu_count(), len(m_values) * n_sequences)

    if logger:
        logger.info(f'模块1: {len(m_values)} m值 x {n_sequences} seq = {len(m_values) * n_sequences} 任务, {n_workers} workers')

    # 构建任务列表
    tasks = []
    for m in m_values:
        for seq in range(n_sequences):
            tasks.append((m, seq, shots, eps1, hash((m, seq)) % 100000))

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
# 模块 2：标定双门错误率 ε₂（并行版，isolated 模式）
# ============================================================================
# 函数功能：对 2 个 qubit 施加 m cycle 单门+ISWAP，计算 F_XEB，拟合得 ε₂
# 并行方式：把 (m, seq) 组合拆成独立任务，用 ProcessPoolExecutor 并行执行
# 任务数 = len(m_values) × n_sequences
# ============================================================================
# 参数说明：
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
def calibrate_two_gate_eps2_parallel(m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS, n_workers=None, eps1=EPS1, eps2=EPS2, logger=None):
    if n_workers is None:
        n_workers = min(cpu_count(), len(m_values) * n_sequences)

    if logger:
        logger.info(f'模块2: {len(m_values)} m值 x {n_sequences} seq = {len(m_values) * n_sequences} 任务, {n_workers} workers')

    # 构建任务列表
    tasks = []
    for m in m_values:
        for seq in range(n_sequences):
            tasks.append((m, seq, shots, eps1, eps2, hash((m, seq)) % 100000))

    # 并行执行
    f_xeb_by_m = {m: [] for m in m_values}

    completed = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_two_gate_experiment, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                m, f_xeb = future.result()
                f_xeb_by_m[m].append(f_xeb)
                completed += 1
                if logger:
                    logger.info(f'[进度] 模块2 [{completed}/{total}] m={m}, seq={task[1]}, F_XEB={f_xeb:.6f}')
            except Exception as e:
                if logger:
                    logger.error(f'模块2任务失败: {e}')

    # 取平均
    f_xeb_avg_by_m = {m: np.mean(f_xeb_by_m[m]) for m in m_values}

    # 拟合
    m_array = np.array(list(f_xeb_avg_by_m.keys()))
    f_xeb_array = np.array(list(f_xeb_avg_by_m.values()))

    def f_xeb_model(m, eps):
        D = 4
        return (1 - eps / (1 - 1.0/D**2)) ** m

    try:
        popt, _ = curve_fit(f_xeb_model, m_array, f_xeb_array, p0=[0.01], bounds=(0, 1))
        eps2_c = popt[0]
    except RuntimeError:
        eps2_c = 0.01

    # ε₂ = ε₂ᶜ - 2ε₁
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
    parser = argparse.ArgumentParser(description='XEB 错误率标定实验（isolated + Neill）')
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

    logger.info(f'参数: m_values={m_values}, n_sequences={args.n_sequences}, shots={args.shots}')
    logger.info(f'n_workers: {args.n_workers or cpu_count()}')

    # =========================================================================
    # 端点验证：无噪声时（EPS1=EPS2=0），F_XEB ≈ 1
    # =========================================================================
    logger.info('[验证] 无噪声端点测试 (EPS1=EPS2=0)...')
    # 单门：m=3, 无噪声（用 np.random.seed(42) 固定随机性）
    np.random.seed(42)
    test_circuit = Circuit(1)
    for _ in range(3):
        gate = np.random.choice(SINGLE_GATES)
        if gate == 'sqrtX':
            params = [np.pi/2, -np.pi/2, np.pi/2]
        elif gate == 'sqrtY':
            params = [np.pi/2, 0, 0]
        elif gate == 'sqrtW':
            params = [np.pi/2, -np.pi/4, np.pi/4]
        test_circuit.add_gate('U3', 0, params=params)
    test_circuit.measure(0)
    ideal_sim = Simulator(backend_type='statevector')
    psi = ideal_sim.simulate_statevector(test_circuit)
    p_exp = np.abs(np.array(psi)) ** 2
    # 无噪声模拟：p_meas = p_exp
    f_xeb_single = compute_fxeb_neill(p_exp, p_exp, 1)
    logger.info(f'[验证] 单门：P_meas = P_exp → F_XEB = {f_xeb_single:.6f} (期望 ≈ 1.0)')

    # 双门：m=3, 无噪声
    np.random.seed(42)
    test_circuit2 = Circuit(2)
    for _ in range(3):
        # q0
        gate0 = np.random.choice(SINGLE_GATES)
        if gate0 == 'sqrtX':
            params0 = [np.pi/2, -np.pi/2, np.pi/2]
        elif gate0 == 'sqrtY':
            params0 = [np.pi/2, 0, 0]
        elif gate0 == 'sqrtW':
            params0 = [np.pi/2, -np.pi/4, np.pi/4]
        test_circuit2.add_gate('U3', 0, params=params0)
        # q1
        gate1 = np.random.choice(SINGLE_GATES)
        if gate1 == 'sqrtX':
            params1 = [np.pi/2, -np.pi/2, np.pi/2]
        elif gate1 == 'sqrtY':
            params1 = [np.pi/2, 0, 0]
        elif gate1 == 'sqrtW':
            params1 = [np.pi/2, -np.pi/4, np.pi/4]
        test_circuit2.add_gate('U3', 1, params=params1)
        # ISWAP
        test_circuit2.iswap(0, 1)
    test_circuit2.measure(0)
    test_circuit2.measure(1)
    psi2 = ideal_sim.simulate_statevector(test_circuit2)
    p_exp2 = np.abs(np.array(psi2)) ** 2
    f_xeb_double = compute_fxeb_neill(p_exp2, p_exp2, 2)
    logger.info(f'[验证] 双门：P_meas = P_exp → F_XEB = {f_xeb_double:.6f} (期望 ≈ 1.0)')

    # 均匀分布测试：P_meas = uniform → F_XEB = 0
    D_single = 2
    p_uniform_single = np.full(D_single, 1.0/D_single)
    f_xeb_uniform_single = compute_fxeb_neill(p_uniform_single, p_exp, 1)
    logger.info(f'[验证] 单门：P_meas = uniform → F_XEB = {f_xeb_uniform_single:.6f} (期望 ≈ 0.0)')

    D_double = 4
    p_uniform_double = np.full(D_double, 1.0/D_double)
    f_xeb_uniform_double = compute_fxeb_neill(p_uniform_double, p_exp2, 2)
    logger.info(f'[验证] 双门：P_meas = uniform → F_XEB = {f_xeb_uniform_double:.6f} (期望 ≈ 0.0)')

    # =========================================================================
    # 模块1
    # =========================================================================
    logger.info('[模块1] 开始...')
    eps1 = calibrate_single_gate_eps1_parallel(
        m_values=m_values,
        n_sequences=args.n_sequences,
        shots=args.shots,
        n_workers=args.n_workers,
        eps1=args.eps1,
        logger=logger
    )
    logger.info(f'[模块1] ε₁ = {eps1:.6f}')

    # =========================================================================
    # 模块2
    # =========================================================================
    logger.info('[模块2] 开始...')
    eps2_c, eps2 = calibrate_two_gate_eps2_parallel(
        m_values=m_values,
        n_sequences=args.n_sequences,
        shots=args.shots,
        n_workers=args.n_workers,
        eps1=eps1,
        eps2=args.eps2,
        logger=logger
    )
    logger.info(f'[模块2] ε₂ᶜ = {eps2_c:.6f}, ε₂ = {eps2:.6f}')

    # =========================================================================
    # 模块3
    # =========================================================================
    logger.info('[模块3] 开始...')
    e_0, e_1, e_q = calibrate_readout_eq(n_qubits=1, shots=3000)
    logger.info(f'[模块3] e_0 = {e_0:.6f}, e_1 = {e_1:.6f}, e_q = {e_q:.6f}')

    # =========================================================================
    # 模块4：计算门数并用 Eq.77 预测保真度
    # =========================================================================
    #   G_1 = n*m = 20*20 = 400（单量子门总数）
    G_1 = 400
    #   G_2 = 27*m/4 = 27*20/4 = 135（双量子门总数，ABCD 平均）
    G_2 = 135
    #   n_measurements = n = 20（测量数，每个 qubit 测 1 次）
    n_measurements = 20
    #   F = (1-ε₁)^G₁ · (1-ε₂)^G₂ · (1-e_q)^n_measurements
    F = predict_fidelity_eq77(eps1, eps2, e_q, G_1, G_2, n_measurements)
    logger.info(f'[模块4] F = {F:.6e}')

    # =========================================================================
    # 汇总
    # =========================================================================
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
            f.flush()
            os.fsync(f.fileno())
        logger.info(f'结果已保存到 {args.output}')

    return eps1, eps2_c, eps2, e_q, F


# ============================================================================
# 命令行入口
# ============================================================================
# 用法：
#   python calibrate_v2_pseudocode_2.0_isolated_neill.py                          # 全参数
#   python calibrate_v2_pseudocode_2.0_isolated_neill.py --m-values 1,5       # 小参数
#   python calibrate_v2_pseudocode_2.0_isolated_neill.py --output result.json # 保存结果
# ============================================================================
if __name__ == '__main__':
    main()
