# XEB 实验复现（基于 uniqc 代码库）
# =====================================
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
# 实验参数（20 qubit / 20 cycle / K=10，与 Sycamore 实验一致）：
#   n  = 20        qubit 数
#   m  = 20        cycle 数（ABCDCDAB × 2 + ABCD = 20 cycle）
#   N_s= 10**3     单次实验采样数
#   K  = 10        独立电路重复数
#
# 噪声参数：
#   EPS1 = 0.0025   单量子门 p 值（用于 Depolarizing 演化）
#   EPS2 = 0.0074   双量子门 p 值（用于 TwoQubitDepolarizing 演化）
#   EPS1_ERR = 0.0016  单量子门实际错误率 ε（用于 alpha_f 预测）
#   EPS2_ERR = 0.0062  双量子门实际错误率 ε（用于 alpha_f 预测）
#
# 使用方法：
#   将UNIQC_XXX() 函数替换为uniqc库的真实 API。
#   所有占位符函数当前都会 raise NotImplementedError——替换完才能跑。
#
# F_XEB计算公式：
#   F_XEB = 2^n · Σ_i P_sampled(x_i) · P_expected(x_i) - 1
#
# 命令行：
#   python xeb_reproduce.py            # 无噪声（理想演化，ε=0）
#   python xeb_reproduce.py --noise    # 含噪声（按eps1/eps2加噪声）
#
# 要改噪声大小，直接改源码顶部的 EPS1/EPS2 即可；不要噪声就把 --noise 去掉


# ============================================================
# 实验参数
# ============================================================
# 与 Sycamore 实验一致
N_QUBITS    = 20       # qubit 数
M_CYCLES    = 20       # cycle 数（循环序列：ABCDCDAB）
N_s         = 10 ** 3   # 单次实验采样数（10³）
K_REPEATS   = 10        # 独立电路重复数,即K
EPS1        = 0.0028    # 单量子门 p 值（用于演化）
EPS2        = 0.0071    # 双量子门 p 值（用于演化）
EPS1_ERR    = 0.0016   # 单量子门实际错误率（用于 alpha_f 预测）
EPS2_ERR    = 0.0062   # 双量子门实际错误率（用于 alpha_f 预测）

# ============================================================
# 代码库调用
# ============================================================
import numpy as np
import random
import argparse
import time
from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, ErrorLoader_GateTypeError, NoisySimulator, ErrorLoader_GenericError, TwoQubitDepolarizing

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
# 【模块 1：量子线路构建】步骤 1 — 生成随机线路 U
# ============================================================
# 函数功能：生成 n qubit × m cycle 的随机线路 U
# ============================================================
# 参数说明：
#   n       : 比特数
#   m       : cycle 数
#   seed    : 随机种子（保证线路可复现，即同 seed=相同量子线路；内部作为 random.seed() 的起点）
#   circuit : 返回值，生成的随机量子线路对象（uniqc 线路）
# ============================================================
def generate_random_circuit(n, m, seed=None):
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
            # 4.1.3 施加单门：用 U3 实现 √X, √Y, √W（ 差一个全局相位 exp(iα)，在量子计算中等价，不影响测量结果）
            if gate == 'sqrtX':
                circuit.add_gate('U3', qubit, params=[np.pi/2, -np.pi/2, np.pi/2])
            elif gate == 'sqrtY':
                circuit.add_gate('U3', qubit, params=[np.pi/2, 0, 0])
            elif gate == 'sqrtW':
                circuit.add_gate('U3', qubit, params=[np.pi/2, -np.pi/4, np.pi/4])
            # 4.1.4 记录该cycle使用的单门：更新该 qubit 的"上一 cycle 单门"记录到列表，供下一 cycle 参考
            last_single_gate[qubit] = gate
    #   4.2 双量子门层：按 ABCDCDAB 序列查本 cycle 应使用的子集，遍历子集内所有 qubit 对施加双门
        #   4.2.1 确定本序列字母：将cycle序号对8取模，查 ABCDCDAB 序列得本 cycle 子集对应的字母（A/B/C/D）
        letter = PATTERN_8CYCLE[cycle % 8]
        #   4.2.2 查双量子比特对：查 ABCD_PAIRS 表得本 cycle 要施加的所有 qubit 对
        pairs = ABCD_PAIRS[letter]
        #   4.2.3 施加双门：遍历所有qubit对，只对在 n 范围内的对施加双门
        for (q1, q2) in pairs:
            if q1 < n and q2 < n:
                circuit.iswap(q1, q2)
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
        # 双门噪声加到 iswap 门上
        error_model = ErrorLoader_GateTypeError(
            generic_error=[Depolarizing(p=0)],  # 无默认噪声
            gatetype_error={
                'U3': [Depolarizing(p=EPS1)],          # 单门噪声
                'ISWAP': [TwoQubitDepolarizing(p=EPS2)], # 双门噪声
            }
        )
        # 2.2 创建含噪声模拟器
        sim_noisy  = NoisySimulator(backend_type='statevector', error_loader=error_model)
    # 3.用无噪声模拟器跑线路，得到理想的纯态矢量|ψ⟩
    psi = sim_ideal.simulate_statevector(circuit)
    # 4.根据模拟器类型选择测量方式得到比特串
    #   4.1 若 use_noise=False：无噪声模拟器直接测量 N_s 次，返回比特串列表
    if not use_noise:
        shot_result = sim_ideal.simulate_shots(circuit,shots=N_s)
        bitstrings = []
        for bitstring, count in shot_result.items():
            bitstrings.extend([bitstring] * count)
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
def compute_F_XEB(psi, bitstrings, n):
    # 1.根据 |ψ⟩ 算理论概率 p_expected(x) = |⟨x|ψ⟩|²（每个分量取模方）
    # 注：uniqc 返回的 psi 是 list，需转为 numpy array
    psi = np.array(psi)
    p_expected = np.abs(psi) ** 2
    # 2.根据bitstrings 统计实测概率 p_measured(x) = (比特串 x 出现次数) / N_s
    unique ,counts = np.unique(bitstrings, return_counts=True)
    p_measured = counts / N_s
    # 3.加权和：Σ_i P_measured(x_i) · P_expected(x_i)
    weight_sum = np.sum(p_measured * p_expected[unique])
    # 4.代入公式计算：F_XEB = 2^n · 加权和 - 1
    F_XEB = (2**n) * weight_sum - 1
    # 5.返回 F_XEB
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
def run_repeat_experiment(n, m, use_noise, N_s):
    # 1.初始化结果列表：F_list = []（用于存 K 次实验的 F_XEB）
    F_list = []
    # 2.循环 K_REPEATS 次（k = 0, 1, …, K_REPEATS-1），每次跑 1 个新随机线路：
    for k in range(K_REPEATS):
        #   2.1 生成随机线路：调用模块1的量子线路构建函数，生成第 k 个随机线路
        circuit = generate_random_circuit(n, m, seed=k)

        #   2.2 定义进度回调函数
        last_print = 0
        def progress_callback(current_count, total_count, psi, bitstrings, n_qubits):
            nonlocal last_print
            if current_count // 100 > last_print:
                F_XEB = compute_F_XEB(psi, bitstrings, n_qubits)
                print(f"已采 {current_count}/{total_count}: F_XEB 估计 = {F_XEB:.4f}")
                last_print = current_count // 100

        #   2.3 跑模拟与测量：调用模块2的模拟与测量函数，得到比特串和纯态 psi
        bitstrings, psi = run_circuit(circuit, use_noise, N_s, seed=k, progress_callback=progress_callback)

        #   2.4 计算保真度：调用模块3的保真度计算函数，算本组的 F_XEB
        F_XEB = compute_F_XEB(psi, bitstrings, n)

        #   2.5 记录：把本组 F_XEB 添加到 F_list
        F_list.append(F_XEB)

    # 3.根据 F_list，用 numpy 计算 F_XEB 的均值和标准差
    F_XEB_mean = np.mean(F_list)
    F_XEB_std = np.std(F_list)
    # 4.返回均值和标准差
    return F_XEB_mean, F_XEB_std


# ============================================================
# 【模块 5：主函数】XEB 实验主流程
# ============================================================
# 函数功能：XEB 实验主流程
# ============================================================
# 参数说明：
#   use_noise : 是否加噪声（False=无噪声理想演化；True=含 Depolarizing 噪声演化）
#               噪声大小由顶部的 EPS1 / EPS2 控制
# 返回值：
#   F_mean : K 次 F_XEB 的均值
#   F_std  : K 次 F_XEB 的标准差
#   alpha_f : 理论预测的保真度（用于对比；无噪声时 = 1）
# ============================================================
def main(use_noise=False):
    # 1.打印实验模式（无噪声/含噪）+ 实验参数
    if use_noise:
        print(f"XEB Experiment - NOISY mode")
        print(f"Parameters: n={N_QUBITS}, m={M_CYCLES}, N_s={N_s}, K={K_REPEATS}")
        print(f"Noise: eps1={EPS1}, eps2={EPS2}")
    else:
        print(f"XEB Experiment - IDEAL mode")
        print(f"Parameters: n={N_QUBITS}, m={M_CYCLES}, N_s={N_s}, K={K_REPEATS}")
    # 2.预测保真度 alpha_f（用于判断 N_s 是否足够）：
    #   2.1 算 G_1 = n*m（单量子门总数）、G_2 = 27*m/4（双量子门总数，ABCD 平均）
    G_1 = N_QUBITS * M_CYCLES
    G_2 = 27 * M_CYCLES / 4
    #   2.2 预测保真度 alpha_f（用实际错误率，不是 p 值）
    if use_noise:
        alpha_f = (1-EPS1_ERR)**G_1 * (1-EPS2_ERR)**G_2
    else:
        alpha_f = 1.0
    #   2.3 若 use_noise=True，判断 N_s 是否足够（阈值：N_s ≳ 10/α_f²）：
    if use_noise:
        threshold = 10 / (alpha_f ** 2)
        #   - 若 N_s < threshold：打印预警
        if N_s < threshold:
            print(f"WARNING: N_s={N_s} may be insufficient (threshold={threshold:.2e})")
        #   - 若 N_s ≥ threshold：打印 OK
        else:
            print(f"N_s is sufficient")
    # 3.主实验：调用模块 4 的 run_repeat_experiment 运行 K 次重复，得 (F_mean, F_std)
    F_mean, F_std = run_repeat_experiment(N_QUBITS, M_CYCLES, use_noise, N_s)
    # 4.打印实验结果
    print(f"Result: F_XEB = {F_mean:.6f} +/- {F_std:.6f}")
    print(f"Theory: alpha_f = {alpha_f:.6f}")
    # 5.返回 (F_mean, F_std, alpha_f)
    return F_mean, F_std, alpha_f

# ============================================================
# 【命令行入口】使用案例
#   python xeb_v2.py            # 无噪声
#   python xeb_v2.py --noise    # 含噪声
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", action="store_true", help="启用含噪声模拟")
    args = parser.parse_args()
    main(use_noise=args.noise)
