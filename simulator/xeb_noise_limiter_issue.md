# XEB 实验与 uniqc 噪声模拟器限制问题

## 问题描述

XEB 实验需要 20 qubit，但 uniqc 的噪声模拟器最多只支持 10 qubit。

## 实验需求

xeb_v2.py 需要：
1. **无噪声模拟**：statevector 模拟器 → 理想态 |ψ⟩
2. **含噪声模拟**：density_matrix 模拟器 → 概率分布 → 采样 N_s 个比特串
3. 计算 F_XEB 保真度

参数：20 qubit × 20 cycle，N_s = 10^6，K = 10

## uniqc 限制

### 文档说明
根据 `docs/source/2_advanced/noise_simulation.md`：
> `backend_type` 必须是 `'density_matrix'`，状态向量后端不支持噪声

### 实际测试结果

| qubit 数 | 结果 |
|---------|------|
| ≤10 | ✓ 成功 |
| >10 | ✗ 失败 |

错误信息：
```
ValueError: Exceed max_qubit_num (nqubit = 20, limit = 10)
```

### 源码位置
```
/home/ubuntu/UnifiedQuantum-main/UniqcCpp/src/density_operator_simulator.h:19
static inline size_t max_qubit_num = 10;
```

这是 C++ 源码硬编码，无法通过 Python 参数配置修改。

## 可能的解决方案

1. **减少 qubit 数量**：用 10 qubit 做实验（与 Sycamore 论文不一致）
2. **手动噪声近似**：用 statevector 模拟器 + 手动加噪声（不精确）
3. **修改源码**：增加 max_qubit_num（需要重新编译 C++ 扩展）
4. **其他方案**：有待老师指导

## 相关文件

- xeb_v2.py：主实验代码
- /home/ubuntu/UnifiedQuantum-main/UniqcCpp/src/density_operator_simulator.h：C++ 限制源码
