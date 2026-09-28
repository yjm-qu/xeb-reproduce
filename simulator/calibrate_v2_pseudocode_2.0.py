# XEB 错误率标定实验 v2（复现 Arute 2019 流程）
# =====================================
# 目的：从比特串分布算 XEB fidelity，拟合得 ε₁/ε₂/e_q

# ============================================================================
# 实验参数（全局）
# ============================================================================
N_QUBITS = 20              # qubit 数量
M_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # m 序列
N_SEQUENCES = 10           # 每个 m 跑多少组随机序列
SHOTS = 2000               # 每组采样数
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
from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, NoisySimulator, ErrorLoader_GateTypeError, TwoQubitDepolarizing
from scipy.optimize import curve_fit


# ============================================================================
# 模块 1：标定单门错误率 ε₁
# ============================================================================
# 函数功能：对 n_qubits 个 qubit 施加 m 个随机单门，测 XEB fidelity，拟合得 ε₁
# ============================================================================
# 参数说明：
#   n_qubits    : qubit 数量（默认 20）
#   m_values    : m 序列
#   n_sequences : 每个 m 跑多少组
#   shots       : 每组采样数
# 返回值：
#   eps1 : 拟合得的 ε₁
# ============================================================================
def calibrate_single_gate_eps1(n_qubits=N_QUBITS, m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS):
    # 1.遍历每个 m（10 个 m 值）
    #    for m in m_values:
    #        f_xeb_list = []  # 该 m 下的 n_sequences 组序列结果
    f_xeb_avg_by_m = {}
    for m in m_values:
        # 2.遍历 n_sequences 组随机序列
        f_xeb_list = []
        for seq in range(n_sequences):
            # 2.1 生成含 m 个随机单门的序列（每个 qubit 独立生成）
            # 2.2 构造电路：每个 qubit 依次施加 sqrtX/sqrtY/sqrtW（U3 实现）
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

            # 2.3a 理想 Simulator 跑电路得 P_U(b)
            ideal_sim = Simulator(backend_type='statevector')
            psi = ideal_sim.simulate_statevector(circuit)
            p_u = np.abs(np.array(psi)) ** 2

            # 2.3b ErrorLoader_GateTypeError + Depolarizing(p=EPS1) 噪声模型，NoisySimulator 跑得N_S个量子比特串 samples
            error_model = ErrorLoader_GateTypeError(
                generic_error=[Depolarizing(p=0)],
                gatetype_error={'U3': [Depolarizing(p=EPS1)]}
            )
            noisy_sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
            noisy_result = noisy_sim.simulate_shots(circuit.originir, shots=shots)

            # 2.4 对每个 qubit q marginalize 出 1-bit 分布 P_U^(q) 和 P_noisy^(q)，算每 qubit 的 F_XEB^(q) = 2·Σ -1（D=2），再对 qubit 求平均
            D = 2
            f_xeb_per_qubit = []
            for q in range(n_qubits):
                # marginalize 理想分布得 qubit q 的 1-bit 分布
                p_u_q = np.zeros(2)
                for b in range(len(p_u)):
                    bit_q = (b >> q) & 1
                    p_u_q[bit_q] += p_u[b]
                # marginalize 噪声采样得 qubit q 的 1-bit 分布（计数）
                noisy_q = np.zeros(2)
                for b, count in noisy_result.items():
                    bit_q = (b >> q) & 1
                    noisy_q[bit_q] += count
                # 算该 qubit 的 F_XEB^(q) = 2·Σ P_U^(q)·P_noisy^(q) - 1
                weight_sum_q = sum(p_u_q[bs] * noisy_q[bs] for bs in range(2)) / shots
                f_xeb_q = D * weight_sum_q - 1
                f_xeb_per_qubit.append(f_xeb_q)
            # 对所有 qubit 的 F_XEB 求平均得该 m 该 seq 的 f_xeb
            f_xeb = np.mean(f_xeb_per_qubit)

            # 2.5 记录下这一遍实验中的保真度F_XEB取值，f_xeb_list.append(F_XEB)
            f_xeb_list.append(f_xeb)

        # 3.该 m 的 n_sequences 组 F_XEB 取平均 → f_xeb_avg
        f_xeb_avg_by_m[m] = np.mean(f_xeb_list)

    # 4.拟合所有 m 的 (m, f_xeb_avg) → F_XEB(m) = [1 - ε₁/(1-1/D²)]^m, D=2 得 ε₁
    m_array = np.array(list(f_xeb_avg_by_m.keys()))
    f_xeb_array = np.array(list(f_xeb_avg_by_m.values()))

    def f_xeb_model(m, eps):
        D = 2
        return (1 - eps / (1 - 1.0/D**2)) ** m

    popt, _ = curve_fit(f_xeb_model, m_array, f_xeb_array, p0=[0.01], bounds=(0, 1))

    # 5.返回 ε₁
    return popt[0]


# ============================================================================
# 模块 2：标定双门错误率 ε₂（per-layer XEB）
# ============================================================================
# 函数功能：对 20 qubit 施加 m cycle 单门+ISWAP，marginalize 得 2-bit 分布，拟合得 ε₂
# ============================================================================
# 参数说明：
#   n_qubits    : qubit 数量（默认 20）
#   m_values    : m cycle 序列
#   n_sequences : 每个 m 跑多少组
#   shots       : 每组采样数
#   eps1        : 已标的单门错误率（用于拆出纯双门）
# ============================================================================
def calibrate_two_gate_eps2(n_qubits=N_QUBITS, m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS, eps1=None):
    # 1.遍历每个 layer
    # 好处：一次 simulate_shots 返回完整 20-qubit 分布，同时包含所有 pair 的 2-bit 信息
    # 电路模拟次数：4 layer × 10 m × 5 seq = 200 次
    eps2_c_by_layer = {}
    for layer_name, pair_list in LAYER_DEFS.items():
        # 2.初始化 pair_data 容器：{(i,j): {m: [F_XEB for each seq]}}
        pair_data = {(i, j): {m: [] for m in m_values} for (i, j) in pair_list}

        # 3.遍历每个 m
        for m in m_values:
            # 4.遍历 n_sequences 组随机序列
            for seq in range(n_sequences):
                # 4.1 构造 m cycle：单门层 + ISWAP 双门层
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
                    # 4.2 双门用 circuit.ISWAP(q1, q2)
                    for (q1, q2) in pair_list:
                        circuit.iswap(q1, q2)
                # 测量
                for q in range(n_qubits):
                    circuit.measure(q)

                # 4.3 ErrorLoader_GateTypeError: 单门 Depolarizing(p=EPS1), 双门 TwoQubitDepolarizing(p=EPS2)，NoisySimulator 跑得 samples
                error_model = ErrorLoader_GateTypeError(
                    generic_error=[Depolarizing(p=0)],
                    gatetype_error={
                        'U3': [Depolarizing(p=EPS1)],
                        'ISWAP': [TwoQubitDepolarizing(p=EPS2)]
                    }
                )
                noisy_sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
                noisy_result = noisy_sim.simulate_shots(circuit.originir, shots=shots)

                # 4.3a 理想 Simulator 跑同一电路得 P_U
                ideal_sim = Simulator(backend_type='statevector')
                psi = ideal_sim.simulate_statevector(circuit)
                p_u = np.abs(np.array(psi)) ** 2

                # 4.4 遍历该 layer 所有 pair，每个 pair 独立存数据
                for (i, j) in pair_list:
                    # 4.4.1 marginalize samples 和 P_U 得 pair 2-bit 分布
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

                    # 4.4.2 算 2-bit F_XEB = 4·mean(P_U_pair) - 1
                    weight_sum = sum(p_u_pair[bs] * noisy_pair[bs] for bs in range(4)) / shots
                    f_xeb = 4 * weight_sum - 1

                    # 4.4.3 存到 pair_data[(i, j)][m].append(F_XEB_pair)
                    pair_data[(i, j)][m].append(f_xeb)

        # 5.每个 pair 独立拟合（在 seq 循环外、layer 循环内）
        eps2_c_pair = {}
        for (i, j) in pair_list:
            # 5.1 该 pair 的 F_XEB_curve[m] = mean(pair_data[(i,j)][m] for m in m_values)
            f_xeb_curve = [np.mean(pair_data[(i, j)][m]) for m in m_values]
            # 5.2 拟合 F_XEB_curve vs m → ε₂ᶜ_pair
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

        # 6.跨 pair 平均（layer 内）
        eps2_c_layer = np.mean(list(eps2_c_pair.values()))
        eps2_c_by_layer[layer_name] = eps2_c_layer

    # 7.跨 layer 平均 → ε₂ᶜ
    eps2_c = np.mean(list(eps2_c_by_layer.values()))

    # 8.ε₂ = ε₂ᶜ - 2ε₁
    if eps1 is None:
        eps1 = EPS1
    eps2 = eps2_c - 2 * eps1

    # 9.返回 ε₂ᶜ, ε₂
    return eps2_c, eps2


# ============================================================================
# 模块 3：标定读出错误率 e_q
# ============================================================================
# 函数功能：制备已知态 |0⟩/|1⟩，测量比对得读出错误率
# ============================================================================
# 参数说明：
#   n_qubits : qubit 数量
#   shots    : 每种态采样数
# ============================================================================
def calibrate_readout_eq(n_qubits=1, shots=3000):
    # 1.遍历制备态 state in [0, 1]
    results = {}
    for state in [0, 1]:
        # 1.1 制备：circuit.x(0) if state==1 else 无操作
        circuit = Circuit(n_qubits)
        if state == 1:
            circuit.x(0)
        circuit.measure(0)

        # 1.2 测量：simulate_shots(shots)
        sim = Simulator(backend_type='statevector')
        result = sim.simulate_shots(circuit.originir, shots=shots)

        # 1.3 统计错判率：error_count / shots
        correct_count = result.get(state, 0)
        error_count = shots - correct_count
        results[state] = error_count / shots

    # 2.返回 e_0, e_1, e_q_mean
    e_0 = results[0]
    e_1 = results[1]
    e_q_mean = (e_0 + e_1) / 2
    return e_0, e_1, e_q_mean


# ============================================================================
# 模块 4：Eq.77 预测 fidelity
# ============================================================================
# 函数功能：代入 ε₁/ε₂/e_q 预测 F，与实际跑霸权电路对比
# ============================================================================
# 参数说明：
#   eps1           : 单门错误率
#   eps2           : 双门错误率
#   eq             : 读出错误率
#   n_single_gates : 电路中单门数
#   n_two_gates    : 电路中双门数
#   n_measurements : 测量数
# ============================================================================
def predict_fidelity_eq77(eps1, eps2, eq, n_single_gates, n_two_gates, n_measurements):
    # 1.Eq.77: F = (1-ε₁)^n₁ · (1-ε₂)^n₂ · (1-e_q)^n_q
    F = (1 - eps1) ** n_single_gates
    F *= (1 - eps2) ** n_two_gates
    F *= (1 - eq) ** n_measurements

    # 2.返回 F
    return F


# ============================================================================
# 主函数
# ============================================================================
# 函数功能：依次运行模块1→2→3→4，输出 ε₁, ε₂, e_q, F
# ============================================================================
def main():
    # 1.调用模块1标定单门错误率，得 ε₁
    eps1 = calibrate_single_gate_eps1(
        n_qubits=N_QUBITS, m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS
    )
    print(f'[模块1] ε₁ = {eps1:.6f}')

    # 2.调用模块2标定双门错误率（输入 ε₁），得 ε₂ᶜ 和 ε₂
    eps2_c, eps2 = calibrate_two_gate_eps2(
        n_qubits=N_QUBITS, m_values=M_VALUES, n_sequences=N_SEQUENCES, shots=SHOTS, eps1=eps1
    )
    print(f'[模块2] ε₂ᶜ = {eps2_c:.6f}, ε₂ = {eps2:.6f}')

    # 3.调用模块3标定读出错误率，得 e_q（关键诊断点：无噪声模拟器下应 ≈ 0）
    e_0, e_1, e_q = calibrate_readout_eq(n_qubits=1, shots=3000)
    print(f'[模块3] e_0 = {e_0:.6f}, e_1 = {e_1:.6f}, e_q = {e_q:.6f}（无噪声模拟器，预期 = 0）')
    #   诊断：如果 e_q > 0，说明 simulate_shots 在无噪声下也引入了读出错误，需排查

    # 4.调用模块4代入 ε₁, ε₂, e_q 预测保真度 F（验算公式正确性）
    n_single_gates = 1113
    n_two_gates = 430
    n_measurements = 53
    F = predict_fidelity_eq77(eps1, eps2, e_q, n_single_gates, n_two_gates, n_measurements)
    print(f'[模块4] F = {F:.6e}（公式 F=(1-ε₁)^n₁·(1-ε₂)^n₂·(1-e_q)^n_q 的纯验算）')

    # 5.打印汇总
    print(f'\n===== 汇总 =====')
    print(f'ε₁ = {eps1:.6f}, ε₂ᶜ = {eps2_c:.6f}, ε₂ = {eps2:.6f}, e_q = {e_q:.6f}, F = {F:.6e}')

    # 6.返回 eps1, eps2_c, eps2, e_q, F
    return eps1, eps2_c, eps2, e_q, F


# ============================================================================
# 【命令行入口】冒烟测试
# ============================================================================
if __name__ == '__main__':
    main()
