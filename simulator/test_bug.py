# 测试单门和双门是否存在缓存 bug
import sys
sys.path.insert(0, '/home/ubuntu/.venv/lib/python3.10/site-packages')

from uniqc import Circuit, Simulator
from uniqc.simulator import Depolarizing, NoisySimulator, TwoQubitDepolarizing, ErrorLoader_GateTypeError

def test_single_gate():
    """测试单门：同进程内连续调用不同 p"""
    print("=== 单门测试 ===")
    for p in [0.10, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8, 0.90]:
        em = ErrorLoader_GateTypeError(
            generic_error=[Depolarizing(p=0)],
            gatetype_error={'U3': [Depolarizing(p=p)]}
        )
        sim = NoisySimulator(backend_type='statevector', error_loader=em)
        circuit = Circuit(1)
        circuit.add_gate('U3', 0, params=[3.14159, 0, 0])  # X gate
        circuit.measure(0)
        result = sim.simulate_shots(circuit.originir, shots=10000)
        # 期望得到 1（|1⟩）
        error_count = sum(count for bitstring, count in result.items() if bitstring != 1)
        epsilon = error_count / 10000
        print(f"p={p:.3f} → epsilon={epsilon:.4f}  {result}")

def test_two_qubit_gate():
    """测试双门：同进程内连续调用不同 p"""
    print("\n=== 双门测试 ===")
    for p in [0.10, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8, 0.90]:
        em = ErrorLoader_GateTypeError(
            generic_error=[Depolarizing(p=0)],
            gatetype_error={'ISWAP': [TwoQubitDepolarizing(p=p)]}
        )
        sim = NoisySimulator(backend_type='statevector', error_loader=em)
        circuit = Circuit(2)
        circuit.x(0)  # |01⟩ → q0=1, q1=0
        circuit.iswap(0, 1)  # → |10⟩ (编码 2)
        circuit.measure(0)
        circuit.measure(1)
        result = sim.simulate_shots(circuit.originir, shots=10000)
        # 期望得到 2（|10⟩）
        error_count = sum(count for bitstring, count in result.items() if bitstring != 2)
        epsilon = error_count / 10000
        print(f"p={p:.3f} → epsilon={epsilon:.4f}  {result}")

if __name__ == "__main__":
    test_single_gate()
    test_two_qubit_gate()
