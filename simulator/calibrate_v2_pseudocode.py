# XEB 错误率标定实验 v2（复现 Arute 2019 流程）
# =====================================
# 目的：从比特串分布算 XEB fidelity，拟合得 ε₁/ε₂/e_q

# ============================================================================
# 实验参数（全局）
# ============================================================================
N_QUBITS = 20              # qubit 数量
M_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # m 序列
N_SEQUENCES = 5            # 每个 m 跑多少组随机序列
SHOTS = 500                # 每组采样数
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
    #
    #        # 2.遍历 n_sequences 组随机序列
    #        for seq in range(n_sequences):
    #            2.1 生成含 m 个随机单门的序列（每个 qubit 独立生成）
    #            2.2 构造电路：每个 qubit 依次施加 sqrtX/sqrtY/sqrtW（U3 实现）
    #            2.3a 理想 Simulator 跑电路得 P_U(b)
    #            2.3b ErrorLoader_GateTypeError + Depolarizing(p=EPS1) 噪声模型，NoisySimulator 跑得N_S个量子比特串 samples
    #            2.4 对每个采样出的量子比特串 b 查 P_U[b]，求算术平均 → F_XEB = 2·mean - 1
    #            2.5 记录下这一遍实验中的保真度F_XEB取值，f_xeb_list.append(F_XEB)
    #
    #        # 3.该 m 的 n_sequences 组 F_XEB 取平均 → f_xeb_avg
    #
    # 4.拟合所有 m 的 (m, f_xeb_avg) → F_XEB(m) = [1 - ε₁/(1-1/D²)]^m, D=2 得 ε₁
    # 5.返回 ε₁


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
    #
    # 1.遍历每个 layer
    #    for layer_name, pair_list in LAYER_DEFS.items():
    #
    #        # 2.初始化 pair_data 容器：{(i,j): {m: [F_XEB for each seq]}}
    #        pair_data = {(i, j): {m: [] for m in m_values} for (i, j) in pair_list}
    #
    #        # 3.遍历每个 m
    #        for m in m_values:
    #
    #            # 4.遍历 n_sequences 组随机序列
    #            for seq in range(n_sequences):
    #                4.1 构造 m cycle：单门层 + ISWAP 双门层
    #                4.2 双门用 circuit.ISWAP(q1, q2)
    #                4.3 ErrorLoader_GateTypeError: 单门 Depolarizing(p=EPS1), 双门 TwoQubitDepolarizing(p=EPS2)，NoisySimulator 跑得 samples
    #                4.3a 理想 Simulator 跑同一电路得 P_U
    #
    #                # 4.4 遍历该 layer 所有 pair，每个 pair 独立存数据
    #                for (i, j) in pair_list:
    #                    4.4.1 marginalize samples 和 P_U 得 pair 2-bit 分布
    #                    4.4.2 算 2-bit F_XEB = 4·mean(P_U_pair) - 1
    #                    4.4.3 存到 pair_data[(i, j)][m].append(F_XEB_pair)
    #
    #        # 5.每个 pair 独立拟合（在 seq 循环外、layer 循环内）
    #        eps2_c_pair = {}
    #        for (i, j) in pair_list:
    #            5.1 该 pair 的 F_XEB_curve[m] = mean(pair_data[(i,j)][m] for m in m_values)
    #            5.2 拟合 F_XEB_curve vs m → ε₂ᶜ_pair
    #
    #        # 6.跨 pair 平均（layer 内）
    #        ε₂ᶜ_layer = mean(ε₂ᶜ_pair over all pairs)
    #        ε₂ᶜ_by_layer[layer_name] = ε₂ᶜ_layer
    #
    # 7.跨 layer 平均 → ε₂ᶜ
    # 8.ε₂ = ε₂ᶜ - 2ε₁
    # 9.返回 ε₂ᶜ, ε₂


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
    #    1.1 制备：circuit.x(0) if state==1 else 无操作
    #    1.2 测量：simulate_shots(shots)
    #    1.3 统计错判率：error_count / shots
    # 2.返回 e_0, e_1, e_q_mean


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
    # 2.返回 F


# ============================================================================
# 主函数
# ============================================================================
# 函数功能：依次运行模块1→2→3→4，输出 ε₁, ε₂, e_q, F
# ============================================================================
def main():
    # 1.调用模块1标定单门错误率，得 ε₁
    # 2.调用模块2标定双门错误率（输入 ε₁），得 ε₂ᶜ 和 ε₂
    # 3.调用模块3标定读出错误率，得 e_q
    # 4.调用模块4代入 ε₁, ε₂, e_q 预测保真度 F
    # 5.打印 ε₁, ε₂ᶜ, ε₂, e_q, F
    # 6.返回 eps1, eps2, e_q, F
