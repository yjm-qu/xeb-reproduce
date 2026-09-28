# uniqc 库结构分析（供讨论用）

## 1. 整体架构

```
uniqc/
├── simulator/          # 量子模拟器（你主要用的部分）
├── algorithms/         # 算法（VQE, QAOA, HVA等）
├── backend_adapter/    # 后端适配器
├── circuit_builder/    # 电路构建器
├── compile/           # 编译相关
├── calibration/       # 校准
├── gateway/           # 网关
├── qem/               # 量子错误缓解
├── visualization/     # 可视化
└── test/              # 测试
```

## 2. simulator 模块（你用到的部分）

### 2.1 核心类

| 类名 | 用途 |
|------|------|
| `Simulator` | 基础模拟器 |
| `NoisySimulator` | 含噪声模拟器 |
| `StatevectorSimulator` | 态矢量模拟器 |
| `MPSSimulator` | MPS 模拟器 |
| `DensityOperatorSimulator` | 密度矩阵模拟器 |

### 2.2 噪声模型（error_model.py）

| 类名 | 用途 |
|------|------|
| `Depolarizing` | 去极化信道（单比特） |
| `TwoQubitDepolarizing` | 双比特去极化信道 |
| `AmplitudeDamping` | 振幅阻尼 |
| `BitFlip` | 位翻转 |
| `PhaseFlip` | 相位翻转 |
| `PauliError1Q` | 单比特泡利错误 |
| `PauliError2Q` | 双比特泡利错误 |
| `ThermalRelaxation` | 热弛豫 |

### 2.3 错误加载器

| 类名 | 用途 |
|------|------|
| `ErrorLoader_GenericError` | 通用错误（所有门） |
| `ErrorLoader_GateTypeError` | 按门类型加载错误 |
| `ErrorLoader_GateSpecificError` | 按特定门加载错误 |

## 3. 你用到的 API

### 创建电路
```python
from uniqc import Circuit
circuit = Circuit(n_qubits)
circuit.add_gate('U3', 0, params=[theta, phi, lambda])
circuit.ISWAP(0, 1)
circuit.measure(0)
```

### 含噪声模拟
```python
from uniqc.simulator import NoisySimulator, Depolarizing, TwoQubitDepolarizing, ErrorLoader_GateTypeError

error_model = ErrorLoader_GateTypeError(
    generic_error=[Depolarizing(p=0)],
    gatetype_error={
        'U3': [Depolarizing(p=0.01)],
        'ISWAP': [TwoQubitDepolarizing(p=0.02)]
    }
)
sim = NoisySimulator(backend_type='statevector', error_loader=error_model)
result = sim.simulate_shots(circuit.originir, shots=1000)
```

## 4. 已知的 Bug

**TwoQubitDepolarizing 缓存问题**：
- 位置：simulator/error_model.py 或 native 层
- 现象：同进程内连续使用不同 p 值时，p 参数被缓存
- 影响：双比特门噪声参数无法动态修改

---

## 5. 讨论建议

老师可能会问的问题：
1. **设计选择**：为什么用 ErrorLoader 而非直接传噪声模型？
2. **性能**：native 实现 vs Python 实现的区别
3. **扩展性**：如何添加新的噪声模型？
4. **缓存机制**：ErrorLoader 内部是否有缓存逻辑？

---

## 6. 快速上手建议

如果老师允许你修改仓库，你可能需要关注：
1. `simulator/error_model.py` - 噪声模型定义
2. `simulator/simulator.py` - 模拟器核心
3. `__init__.py` - 导出接口
