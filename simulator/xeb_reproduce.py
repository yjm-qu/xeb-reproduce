# XEB 实验复现（基于 uniqc 代码库）
# =====================================
# 5 步实验流程：
#   步骤 1   ：生成随机线路 U（n qubit × m cycle）
#   步骤 2.1 ：无噪声演化（state vector 模拟器）→ 纯态 |ψ⟩
#   步骤 2.2 ：含噪声演化（density matrix 模拟器 + Depolarizing）→ 密度矩阵 ρ
#   步骤 3   ：采样 N_s 个比特串
#   步骤 4   ：算 F_XEB
#   步骤 5   ：重复 K 次取期望和标准差
#
# 实现所需的6个功能模块：
# 模块 1：量子线路构建
# 模块 2：模拟器（包括无噪声与含噪声两种）
# 模块 3：噪声模型
# 模块 4：采样与测量
# 模块 5：保真度F_XEB计算
# 模块 6：重复实验，取K个不同的随机线路
#
# 实验参数（20 qubit / 20 cycle / K=10，与 Sycamore 实验一致）：
#   n  = 20        qubit 数
#   m  = 20        cycle 数（ABCDCDAB × 2 + ABCD = 20 cycle）
#   N_s= 10**6     单次实验采样数
#   K  = 10        独立电路重复数
#
# 量子门错误率（与 Sycamore 实验一致）：
#   eps1 = 0.0016  单量子门错误率
#   eps2 = 0.0062  双量子门错误率
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
N_SAMPLES   = 10 ** 6  # 单次实验采样数
K_REPEATS   = 10       # 独立电路重复数
EPS1        = 0.0016   # 单量子门错误率
EPS2        = 0.0062   # 双量子门错误率

# ============================================================
# 代码库调用
# ============================================================
import numpy as np
import random
import argparse
import time
from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, ErrorLoader_GateTypeError, NoisySimulator
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
def generate_random_circuit(n, m, seed=None):
    # ============================================================
    # 函数功能：生成 n qubit × m cycle 的随机线路 U
    # ============================================================
    # 参数说明：
    #   n       : 比特数
    #   m       : cycle 数
    #   seed    : 随机种子（保证线路可复现，即同 seed=相同量子线路；内部作为 random.seed() 的起点）
    #   circuit : 返回值，生成的随机量子线路对象（uniqc 线路）
    # ============================================================
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
            #       4.1.3 施加单门：调用 uniqc 库接口，把单门施加到 qubit
            circuit.add_gate(gate,qubit)
            #       4.1.4 记录该cycle使用的单门：更新该 qubit 的"上一 cycle 单门"记录到列表，供下一 cycle 参考
            last_single_gate[qubit] = gate
    #   4.2 双量子门层：按 ABCDCDAB 序列查本 cycle 应使用的子集，遍历子集内所有 qubit 对施加双门
        #   4.2.1 确定本序列字母：将cycle序号对8取模，查 ABCDCDAB 序列得本 cycle 子集对应的字母（A/B/C/D）
        letter = PATTERN_8CYCLE[cycle % 8]
        #   4.2.2 查双量子比特对：查 ABCD_PAIRS 表得本 cycle 要施加的所有 qubit 对
        pairs = ABCD_PAIRS[letter]
        #   4.2.3 施加双门：遍历所有qubit对，调用 uniqc 库接口对 (q1,q2) 施加双量子门
        for (q1, q2) in pairs:
            circuit.add_gate('iSWAP',[q1, q2])
    # 5.返回生成的量子线路对象 circuit
    return circuit

# ============================================================
#【模块 2：模拟器 + 模块 3：噪声模型】步骤 2.1 + 2.2 — 演化（无噪声 / 含噪声）
# ============================================================
# 本模块计划采用3个函数实现，其逻辑关系如下图所示：
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │   两种分支：                                                  │
#   │   run_ideal : 无噪声模拟器，无噪声                             │
#   │                → 返回 纯态矢量 |ψ⟩，复数数组，2ⁿ 维列向量       │
#   │   run_noisy : 含噪声模拟器（depolarizing 噪声）                │
#   │                → 返回 含噪密度矩阵 ρ，复数矩阵 ，2ⁿ × 2ⁿ 方阵   │
#   │                （需 eps1/eps2 构造噪声模型）                   │
#   └──────────────────────────────────────────────────────────────┘
#                │                                 │
#                ▼                                 ▼
#                ┌─────────────────────┐ ┌──────────────────────────┐
#                │ run_ideal           │ │ run_noisy                │
#                │ 无噪声演化           │ │ 含噪声演化               │
#                └─────────────────────┘ └────────────┬─────────────┘
#                                                     │ 调用
#                                                     ▼
#                                       ┌─────────────────────────────┐
#                                       │ build_depolarizing_model    │
#                                       │ 组装 depolarizing 噪声模型   │
#                                       └─────────────────────────────┘

def run_ideal(circuit, n):
    # ============================================================
    # 函数功能：调用模拟器，模拟无噪声演化，输出纯态矢量|ψ⟩ = U|0⟩^n
    # ============================================================
    # 参数说明：
    #   circuit : 量子线路对象（来自 generate_random_circuit）
    #   n       : 比特数
    #   psi     : 返回值，纯态矢量|ψ⟩，复数数组，2ⁿ 维列向量
    # ============================================================
    # 1.调用 uniqc 库接口，创建无噪声模拟器（statevector 模拟器默认初态从 |0⟩^n 开始，无需显式制备）
    sim = Simulator('statevector')
    # 2.调用 uniqc 库接口，用无噪声模拟器跑线路，得到纯态矢量 |ψ⟩
    psi = sim.simulate_statevector(circuit)
    # 3.返回纯态矢量 psi(公式中的|ψ⟩)
    return psi

def build_depolarizing_model(eps1, eps2):
    # ============================================================
    # 函数功能：调用 uniqc 库接口，构建 Depolarizing 噪声模型（单门 ε₁、双门 ε₂），数学定义 E(ρ) = (1-ε)·ρ + ε·I/N
    # ============================================================
    # 参数说明：
    #   eps1        : 单量子门错误率 ε₁
    #   eps2        : 双量子门错误率 ε₂
    #   noise_model : 返回值，打包好的噪声模型对象
    # ============================================================
    # 1.调用 uniqc 库接口，创建单量子门噪声通道（错误率 ε₁，作用于 1 qubit）
    single_error = Depolarizing(p = eps1)
    # 2.调用 uniqc 库接口，创建双量子门噪声通道（错误率 ε₂，作用于 2 qubit）
    double_error = Depolarizing(p = eps2)
    # 3.调用 uniqc 库接口，把两个通道按qubit数打包成 1 个噪声模型（{1: 单门通道, 2: 双门通道}）
    noise_model = ErrorLoader_GateTypeError(
        generic_error=[],
        gatetype_error={
            'single':[single_error],
            'double':[double_error],
        }
    )
    # 4.返回噪声模型对象 noise_model
    return noise_model

def run_noisy(circuit, n, eps1, eps2):
    # ============================================================
    # 函数功能：调用模拟器，模拟含噪声演化，输出密度矩阵 ρ（用 density matrix 模拟器 + depolarizing 噪声跑线路）
    # ============================================================
    # 参数说明：
    #   circuit : 量子线路对象（来自 generate_random_circuit）
    #   n       : 比特数
    #   eps1    : 单量子门错误率 ε₁
    #   eps2    : 双量子门错误率 ε₂
    #   rho     : 返回值，含噪密度矩阵ρ，复数矩阵 ，2ⁿ × 2ⁿ 方阵
    # ============================================================
    # 1.调用 build_depolarizing_model，构建 depolarizing 噪声模型（单门 ε₁、双门 ε₂）
    error_model = build_depolarizing_model(eps1, eps2)
    # 2.调用 uniqc 库接口，创建含噪声模拟器（density_matrix 模拟器默认初态从 |0⟩^n 开始，无需显式制备）
    noisy_sim = NoisySimulator('density_matrix', error_loader=error_model)
    # 3.调用 uniqc 库接口，用含噪声模拟器跑线路，传入噪声模型，得到含噪密度矩阵 ρ
    rho = noisy_sim.simulate_density_matrix(circuit)
    # 4.返回含噪声的密度矩阵 rho(公式中的ρ)
    return rho

# ============================================================
#【模块 4：采样与测量】步骤 3 — 采样 N_s 个比特串
# ============================================================
def sample_bitstrings(circuit, n, N_s, seed=None):
    # ============================================================
    # 函数功能：调用 uniqc 库接口，对量子线路 circuit 模拟测量 N_s 次
    # ============================================================
    # 参数说明：
    #   circuit    : 量子线路（来自 generate_random_circuit，已含门操作）
    #   n          : qubit 数
    #   N_s        : 测量次数 N_s（实验中 N_s = 10^6）
    #   seed       : 随机种子（保证随机线路可复现，即同 seed=相同量子线路；内部作为 random_seed 参数传给 uniqc 测量接口）
    #   bitstrings : 返回值，shape=(N_s,)，整数数组，每个值是比特串的整数编码 0 ~ 2ⁿ-1
    # ============================================================
    # 1.在线路的每个 qubit 上添加测量门
    for i in range(n):
        circuit.measure(i)
    # 2.调用 uniqc 库接口，对线路模拟测量 N_s 次，得到比特串数组（bitstrings）
    sim = Simulator('statevector')
    result = sim.simulate_shots(circuit,shots=N_s)
    # 3.将测量结果转换为 bitstrings 数组
    bitstrings = []
    for bitstring,count in result.items():
        bitstrings.extend([bitstring] * count)
    # 4.返回比特串数组 bitstrings
    return np.array(bitstrings)


# ============================================================
# 【模块 5：保真度 F_XEB 计算】步骤 4 — 算 F_XEB
# ============================================================
def compute_F_XEB(psi, bitstrings, n):
    # ============================================================
    # 函数功能：算 F_XEB
    # ============================================================
    # 计算公式：
    #   p_expected(x) = |⟨x|ψ⟩|²（理论概率，从 |ψ⟩ 算）
    #   p_measured(x) = (比特串 x 在 bitstrings 中出现次数) / N_s（实测频率，从 bitstrings 统计）
    #   F_XEB = 2^n · Σ_i P_measured(x_i) · P_expected(x_i) - 1（加权和形式）
    # ============================================================
    # 参数说明：
    #   psi        : 纯态矢量|ψ⟩（来自 run_ideal）—— 用于算 p_expected
    #   bitstrings : shape=(N_s,) 整数数组，比特串的整数编码 0 ~ 2ⁿ-1（来自 sample_bitstrings），用于算 p_measured
    #   n          : qubit 数
    #   F_XEB      : 返回值，该随机线路的保真度
    # ============================================================
    # 1.根据 |ψ⟩ 算理论概率 p_expected(x) = |⟨x|ψ⟩|²（每个分量取模方）
    p_expected = np.abs(psi) ** 2
    # 2.根据bitstrings 统计实测概率 p_measured(x) = (比特串 x 出现次数) / N_s
    N_s = len(bitstrings)
    unique ,counts = np.unique(bitstrings, return_counts=True)
    p_measured = counts / N_s
    # 3.加权和：Σ_i P_measured(x_i) · P_expected(x_i)
    weight_sum = np.sum(p_measured * p_expected[unique])
    # 4.代入公式计算：F_XEB = 2^n · 加权和 - 1
    F_XEB = (2**n) * weight_sum - 1
    # 5.返回 F_XEB
    return F_XEB

# ============================================================
# 【模块 6：重复实验】步骤 5 — 重复 K 次取期望和标准差
# ============================================================
def repeat_K_times(K, n, m, N_s, use_noise, eps1, eps2):
    # ============================================================
    # 函数功能：重复 K 组不同随机线路，取 F_XEB 的期望和标准差
    # ============================================================
    # 参数说明：
    #   K         : 独立电路重复数（实验中 K = 10）
    #   n         : qubit 数
    #   m         : cycle 数
    #   N_s       : 单次实验采样数（实验中 N_s = 10^6）
    #   use_noise : 是否加噪声（False=无噪声；True=含噪声）
    #   eps1      : 单量子门错误率 ε₁（use_noise=True 时生效）
    #   eps2      : 双量子门错误率 ε₂（use_noise=True 时生效）
    #   F_mean    : 返回值，K 次 F_XEB 的均值
    #   F_std     : 返回值，K 次 F_XEB 的标准差
    # ============================================================
    # 1.初始化结果列表：F_list = []（用于存 K 次实验的不同F_XEB）
    # 2.循环 K 次（k = 0, 1, …, K-1），每次跑 1 个新随机线路：
    #   2.1 生成随机线路：调用 模块1的函数generate_random_circuit(n, m, seed=k)，生成第 k 个随机线路（不同 seed=不同电路，保证 K 组相互独立）
    #   2.2 无条件跑无噪声演化：调用 模块2的函数run_ideal(circuit, n)，得到纯态矢量psi（公式中的|ψ⟩）
    #        （先算 psi的理由：函数compute_F_XEB(psi, ...) 的输入参数必须使用纯态矢量 psi(公式中的|ψ⟩)，
    #         因此无论 use_noise 与否，都必须先跑 run_ideal 拿到 psi）
    #   2.3 直接使用原始线路circuit（用作 sample_bitstrings 的输入）
    #   2.4 采样与测量：调用 模块4的函数sample_bitstrings(circuit, n, N_s, seed=k)，从线路模拟测量 N_s 次得到比特串（seed=k 保证可复现）
    #   2.5 计算保真度F_XEB:调用 模块5的函数compute_F_XEB(psi, bitstrings, n)，算本组的 F_XEB
    #   2.6 记录：把本组 F_XEB 添加到 F_list，并打印本组结果（格式 `[k=k+1/K] F_XEB = ...`）
    # 3.根据F_list，算 F_XEB 的期望和标准差
    # 4.返回 (F_mean, F_std)


# ============================================================
# 【主入口】main() — XEB 实验流程总控
# ============================================================
def main(use_noise=False):
    # ============================================================
    # 函数功能：XEB 实验主流程
    # ============================================================
    # 参数说明：
    #   use_noise : 是否加噪声（False=无噪声理想演化；True=含 Depolarizing 噪声演化）
    #               噪声大小由顶部的 EPS1 / EPS2 控制
    #   F_mean    : 返回值，K 次 F_XEB 的均值
    #   F_std     : 返回值，K 次 F_XEB 的标准差
    #   alpha_f   : 返回值，理论预测的保真度（含噪路径下供对比；无噪声时 = 1）
    # ============================================================
    # 1.从文件顶部常量取出本实验参数：n, m, N_s, K, eps1, eps2
    # 2.打印实验模式（无噪声/含噪）+ 实验参数
    # 3.预测保真度 α_f（用于+ 决定 N_s 是否足够）：
    #   3.1 算 G_1 = n*m（单量子门总数）、G_2 = 27*m/4（双量子门总数，ABCD 平均）
    #   3.2 算 alpha_f = (1-eps1)^G_1 * (1-eps2)^G_2
    #   3.3 若 use_noise=True，执行下面的判定
    #     3.3.1 判断N_s 是否足够（阈值：N_s ≳ 10/α_f²（α_f² 量级）：
    #          计算阈值 threshold = 10 / alpha_f**2
    #          - 若 N_s < threshold：打印预警 ⚠ N_s={N_s} 可能不足以分辨 α_f（threshold={threshold:.2e}）
    #          - 若 N_s ≥ threshold：打印 OK，N_s 充足
    # 4.主实验：调用 模块6的函数repeat_K_times(K, n, m, N_s, use_noise, eps1, eps2) 跑 K 次重复，得 (F_mean, F_std)
    # 5.打印实验结果
    # 6.返回 (F_mean, F_std, alpha_f)


# ============================================================
# 【命令行入口】使用案例
#   python xeb_reproduce.py            # 无噪声
#   python xeb_reproduce.py --noise    # 含噪声
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", action="store_true", help="启用含噪声模拟")
    args = parser.parse_args()
    main(use_noise=args.noise)
