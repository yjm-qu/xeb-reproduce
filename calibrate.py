# XEB 模拟器标定实验
# =====================================
# 目的：建立 Depolarizing(p) 的 p 值与实际门错误率 ε 的对应关系
#
# 去极化概率 (depolarizing probability) 解释：
#   - 去极化是一种量子噪声信道
#   - 公式：ρ → (1-p)ρ + p·I/d
#   - p 是去极化概率：p=0 表示无噪声，p=1 表示完全随机（最大噪声）
#   - d 是维度：单比特 d=2，双比特 d=4
#   - 注意：p 不是直接的门错误率，需要通过实验标定
#
# 理论关系（用于验证 uniqc 实现是否与理论一致）：
#   - 单比特: Gate Fidelity ≈ 1 - p/2 → ε ≈ p/2 → p/ε ≈ 2.0
#   - 双比特: Gate Fidelity ≈ 1 - 3p/4 → ε ≈ 3p/4 → p/ε ≈ 1.333
#   - 为什么要"建立关系"而不直接用公式？因为要验证 uniqc 的 Depolarizing 实现是否与理论一致

# ============================================================
# 物理实验流程（单门）- calibrate_single_gate 函数实现
# ============================================================
#   步骤 0：定义噪声模型 → Depolarizing(p=p)
#   步骤 1：制备初态 |0⟩
#   步骤 2：施加单门 X（理想情况：X|0⟩ = |1⟩）
#   步骤 3：用含噪声模拟器模拟 N 次，每次采样得到一个比特串
#   步骤 4：统计错误率 → 理想应得到 |1⟩（编码 1），统计不是 1 的比例 → 这就是 ε
#   【关键】在步骤 0 定义噪声模型时加入去极化概率 p：Depolarizing(p=p)

# ============================================================
# 物理实验流程（双门）- calibrate_two_qubit_gate 函数实现
# ============================================================
#   注意：iSWAP|00⟩ = |00⟩（不变），不能用 |00⟩ 初态
#   正确方法：用 |01⟩ 初态，iSWAP|01⟩ = i|10⟩（交换到 |10⟩）
#   步骤 0：定义噪声模型 → TwoQubitDepolarizing(p=p)，单门无噪声(p=0)
#   步骤 1：制备初态 |01⟩（|00⟩ → X(q0) → |01⟩，q1=第一个 qubit，q0=第二个 qubit）
#   步骤 2：施加双门 iSWAP（理想情况：iSWAP|01⟩ = i|10⟩，相位 i 不影响测量）
#   步骤 3：用含噪声模拟器模拟 N 次，每次采样得到一个比特串
#   步骤 4：统计错误率 → 理想应得到 |10⟩（编码 2，q1 高位 q0 低位，uniqc 固定用大端序），统计不是 2 的比例 → 这就是 ε
#   【关键】在步骤 0 定义噪声模型时加入去极化概率 p：TwoQubitDepolarizing(p=p)
#   【注意】制备 |01⟩ 的 X 门必须设为无噪声(p=0)，否则初态不纯，错误率偏高

# ============================================================
# 命令行使用说明
# ============================================================
# 参数说明：
#   --p      : 单个 depolarizing probability
#              类型: float，范围: 0.0 ~ 1.0，默认: None
#              触发: --p 0 → 场景 1；--p > 0 → 场景 3
#   --p_list : 多个 p 值，用逗号分隔
#              类型: str（如 "0.0001,0.0005,0.001"），默认: None
#              触发: 场景 2
#   --shots  : 测量次数
#              类型: int，范围: > 0，默认: 10000
#   --gate   : 门类型
#              类型: str，默认: 'X'
#              可选: 'X'（单门）、'iswap'（双门）
#
# 参数校验规则：
#   1. --p 和 --p_list 互斥，不能同时给
#   2. 都不给 → 报错（提示用法）
#   3. 都给 → 报错
#   4. --p < 0 或 --p > 1 → 报错（超出范围）
#   5. --shots <= 0 → 报错
#   6. --gate 不是 'X' 或 'iswap' → 报错
#
# 触发逻辑：
#   场景 A（只有 --p 0）       → 场景 1（sanity check）
#   场景 B（只有 --p > 0）     → 场景 1 + 场景 3
#   场景 C（只有 --p_list）    → 场景 1 + 场景 2
#   场景 D（--p 和 --p_list）  → 报错
#   场景 E（什么都不给）       → 报错
#   【注意】场景 1 永远先跑（验证代码逻辑对）
#
# 输出格式：
#   场景 1（sanity check）:
#       输出: p=0.0 → epsilon=0.0000 (PASSED/FAILED)
#       判定: epsilon < 0.001 → PASSED，否则 FAILED
#   场景 2（多 p 扫描）:
#       输出: CSV 格式（不带 # 号），4 位小数
#         p,epsilon
#         0.0008,0.0004
#         0.0012,0.0006
#   场景 3（目标 p 标定）:
#       输出: p=0.0016 → epsilon=0.0008; p/epsilon=2.0000
#       验证: 单门 p/ε≈2.0，双门 p/ε≈1.333
#
# 使用示例：
#   # 1. Sanity check（验证代码逻辑对）
#   python calibrate.py --p 0 --gate X          # 单门 sanity
#   python calibrate.py --p 0 --gate iswap      # 双门 sanity
#
#   # 2. 多 p 扫描（验证 ε-p 线性关系）
#   python calibrate.py --p_list 0.0008,0.0012,0.0016,0.0020,0.0024 --gate X      # 单门
#   python calibrate.py --p_list 0.004,0.005,0.0062,0.007,0.008 --gate iswap      # 双门
#
#   # 3. 目标 p 标定（验证理论 p/ε 比值）
#   python calibrate.py --p 0.0016 --gate X          # 单门，期望 p/ε≈2.0
#   python calibrate.py --p 0.0062 --gate iswap     # 双门，期望 p/ε≈1.333
#
#   # 4. 自定义 shots
#   python calibrate.py --p 0.0016 --shots 100000 --gate X


# ============================================================
# 实验参数
# ============================================================
# 【模块 1：参数预设】导入必要的库
import argparse
import numpy as np
from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, NoisySimulator, TwoQubitDepolarizing, ErrorLoader_GateTypeError


# ============================================================
# 【模块 2：单门标定】标定单门错误率
# ============================================================
# 函数功能：通过实验测量单门在给定 depolarizing p 下的实际错误率 ε
# ============================================================
# 参数说明：
#   p     : depolarizing probability（float，0.0~1.0）
#   gate  : 单门名称（str，默认 'X'）
#   shots : 重复测量次数（int，> 0）
# 返回值：
#   error_rate : 实际测量得到的错误率 ε（float，0.0~1.0，4 位小数）
# ============================================================
def calibrate_single_gate(p, gate='X', shots=10000):
    # 1.定义噪声模型 → Depolarizing(p=p)
    #    【关键】与 xeb_v2.py 保持一致：用 gatetype_error
    #    p=0 时 ε 应该 ≈ 0（验证代码逻辑对）
    # 与 xeb_v2.py 保持一致：用 U3 门
    # U3(π, 0, 0) = X 门，产生确定态 |1⟩
    error_model = ErrorLoader_GateTypeError(
        generic_error=[Depolarizing(p=0)],
        gatetype_error={'U3': [Depolarizing(p=p)]}  # 单门噪声
    )
    # 2.创建含噪声模拟器
    sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
    # 3.构建测试电路 → 制备 |0⟩，施加 U3 门（=X），测量
    circuit = Circuit(1)
    circuit.add_gate('U3', 0, params=[np.pi, 0, 0])  # X 门：|0⟩ → |1⟩
    circuit.measure(0)
    # 4.用含噪声模拟器模拟 shots 次，每次采样得到一个比特串
    shot_result = sim.simulate_shots(circuit.originir, shots=shots)
    bitstrings = []
    for bitstring, count in shot_result.items():
        bitstrings.extend([bitstring] * count)
    # 5.统计错误率 → X|0⟩=|1⟩（编码 1），统计不是 1 的比例
    error_count = sum(1 for b in bitstrings if b != 1)
    error_rate = error_count / shots
    # 6.返回错误率 ε（float，4 位小数）
    return round(error_rate, 4)


# ============================================================
# 【模块 3：双门标定】标定双门错误率
# ============================================================
# 函数功能：通过实验测量双门在给定 depolarizing p 下的实际错误率 ε
# ============================================================
# 参数说明：
#   p     : depolarizing probability（float，0.0~1.0）
#   gate  : 双门名称（str，默认 'iswap'）
#   shots : 重复测量次数（int，> 0）
# 返回值：
#   error_rate : 实际测量得到的错误率 ε（float，0.0~1.0，4 位小数）
# ============================================================
def calibrate_two_qubit_gate(p, gate='iswap', shots=10000):
    # 1.定义噪声模型 → TwoQubitDepolarizing(p=p)，单门无噪声(p=0)
    #    【关键】在这里加入去极化概率 p
    #    注意：制备 |01⟩ 的 X 门必须设为无噪声(p=0)
    #    p=0 时 ε 应该 ≈ 0（验证代码逻辑对）
    error_model = ErrorLoader_GateTypeError(
        generic_error=[Depolarizing(p=0)],  # 单门无噪声
        gatetype_error={
            'ISWAP': [TwoQubitDepolarizing(p=p)]  # 双门加噪声，注意大写
        }
    )
    # 2.创建含噪声模拟器
    sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
    # 3.构建测试电路 → 制备 |01⟩，施加 iSWAP，测量
    #    【注意】不能用 |00⟩ 初态，iSWAP|00⟩ = |00⟩ 不变
    #    正确：用 |01⟩ 初态，iSWAP|01⟩ = i|10⟩ → |10⟩（编码 2，uniqc 固定用大端序）
    circuit = Circuit(2)
    circuit.x(0)  # 制备 |01⟩：q0=1, q1=0（big-endian：q1 高位 q0 低位）
    circuit.iswap(0, 1)  # iSWAP: |01⟩ → i|10⟩
    circuit.measure(0)
    circuit.measure(1)
    # 4.用含噪声模拟器模拟 shots 次，每次采样得到一个比特串
    shot_result = sim.simulate_shots(circuit.originir, shots=shots)
    bitstrings = []
    for bitstring, count in shot_result.items():
        bitstrings.extend([bitstring] * count)
    # 5.统计错误率 → 统计不是 2 的比例（iSWAP|01⟩ = i|10⟩ → |10⟩ 编码为 2）
    error_count = sum(1 for b in bitstrings if b != 2)
    error_rate = error_count / shots
    # 6.返回错误率 ε（float，4 位小数）
    return round(error_rate, 4)


# ============================================================
# 【模块 4：主函数】命令行入口
# ============================================================
# 函数功能：解析命令行参数，运行标定实验
# ============================================================
def main():
    # 1.解析命令行参数
    #    - --p: float，默认 None
    #    - --p_list: str（如 "0.0001,0.0005"），默认 None
    #    - --shots: int，默认 10000
    #    - --gate: str，默认 'X'
    parser = argparse.ArgumentParser(description='XEB 模拟器标定实验')
    parser.add_argument('--p', type=float, default=None, help='单个 depolarizing probability')
    parser.add_argument('--p_list', type=str, default=None, help='多个 p 值，用逗号分隔')
    parser.add_argument('--shots', type=int, default=10000, help='测量次数')
    parser.add_argument('--gate', type=str, default='X', help='门类型：X（单门）或 iswap（双门）')
    args = parser.parse_args()

    # 2.参数校验
    #    - --p 和 --p_list 互斥：都给或都不给 → 报错
    #    - --p 范围检查：< 0 或 > 1 → 报错
    #    - --shots 检查：<= 0 → 报错
    #    - --gate 检查：不是 'X' 或 'iswap' → 报错
    if args.p is not None and args.p_list is not None:
        raise ValueError('--p 和 --p_list 不能同时给')
    if args.p is None and args.p_list is None:
        raise ValueError('--p 或 --p_list 必须给一个')
    if args.p is not None and (args.p < 0 or args.p > 1):
        raise ValueError('--p 必须在 0.0 到 1.0 之间')
    if args.shots <= 0:
        raise ValueError('--shots 必须大于 0')
    if args.gate not in ['X', 'iswap']:
        raise ValueError('--gate 必须是 X 或 iswap')

    p = args.p
    p_list = args.p_list
    shots = args.shots
    gate = args.gate

    # 3.根据 --gate 决定调用哪个函数
    #    - 'iswap' → calibrate_two_qubit_gate
    #    - 'X' → calibrate_single_gate（默认）
    if gate == 'iswap':
        calibrate_func = calibrate_two_qubit_gate
    else:
        calibrate_func = calibrate_single_gate

    # 4.执行实验流程
    #    场景 1：Sanity check（永远先跑）
    #      - 调用 calibrate_xxx(p=0, gate, shots)
    #      - 输出: p=0.0 → epsilon=X.XXXX (PASSED/FAILED)
    #      - 判定: epsilon < 0.001 → PASSED，否则 FAILED
    epsilon_0 = calibrate_func(p=0, gate=gate, shots=shots)
    result_0 = 'PASSED' if epsilon_0 < 0.001 else 'FAILED'
    print(f'p=0.0 → epsilon={epsilon_0:.4f} ({result_0})')

    #    场景 2：多 p 扫描（--p_list 触发）
    #      - 遍历 p_list 中的每个 p
    #      - 调用 calibrate_xxx(p, gate, shots)
    #      - 输出 CSV（不带 # 号，4 位小数）:
    #        p,epsilon
    #        0.0008,0.0004
    #        ...
    if p_list is not None:
        p_values = [float(x) for x in p_list.split(',')]
        print('p,epsilon')
        for p_val in p_values:
            epsilon = calibrate_func(p=p_val, gate=gate, shots=shots)
            print(f'{p_val},{epsilon}')

    #    场景 3：目标 p 标定（--p > 0 触发）
    #      - 调用 calibrate_xxx(p, gate, shots)
    #      - 输出: p=X.XXXX → epsilon=X.XXXX; p/epsilon=X.XXXX
    #      - 验证: 单门 p/ε≈2.0，双门 p/ε≈1.333
    if p is not None and p > 0:
        epsilon = calibrate_func(p=p, gate=gate, shots=shots)
        if epsilon > 0:
            p_over_epsilon = p / epsilon
            print(f'p={p:.4f} → epsilon={epsilon:.4f}; p/epsilon={p_over_epsilon:.4f}')
            if gate == 'X':
                print(f'验证: 单门期望 p/ε≈2.0，实际={p_over_epsilon:.4f}')
            else:
                print(f'验证: 双门期望 p/ε≈1.333，实际={p_over_epsilon:.4f}')
        else:
            print(f'p={p:.4f} → epsilon={epsilon:.4f} (epsilon=0，无法计算 p/epsilon)')


if __name__ == "__main__":
    main()
