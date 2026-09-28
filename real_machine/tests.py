#!/usr/bin/env python3
"""
单元测试模块
包含 xeb_v2_real.py 的自测代码
"""

def test_measure_mapping():
    """
    纯函数自测：测量映射置换
    """
    print("\n=== Measure Mapping Unit Test ===")

    # Case 1: identity mapping
    measure_map_1 = {0: 0, 1: 1, 2: 2, 3: 3}
    counts_1 = {5: 10}  # binary 0101 -> int 5

    # Case 2: swap 0<->1
    measure_map_2 = {0: 1, 1: 0, 2: 2, 3: 3}
    counts_2 = {1: 10}  # binary 0001 -> cbit[0]=1 -> should become logical[0]=1 (from cbit[1]) -> int 2

    # Case 3: physical IDs (模拟编译器的物理编号场景)
    # physical = [5, 9, 7, 2], 逻辑0->物理5, 逻辑1->物理9, 逻辑2->物理7, 逻辑3->物理2
    # MEASURE q[5], c[0] -> c[0] 映射到逻辑0
    # MEASURE q[9], c[1] -> c[1] 映射到逻辑1
    # MEASURE q[7], c[2] -> c[2] 映射到逻辑2
    # MEASURE q[2], c[3] -> c[3] 映射到逻辑3
    # map: c[0]=5->0, c[1]=9->1, c[2]=7->2, c[3]=2->3
    measure_map_3 = {0: 0, 1: 1, 2: 2, 3: 3}  # 归一化后应该是恒等
    physical = [5, 9, 7, 2]
    # 模拟 originir 中 MEASURE q[5], c[0] 等 -> 原始 map 是 {0:5,1:9,2:7,3:2}
    raw_map_3 = {0: 5, 1: 9, 2: 7, 3: 2}
    # 归一化：5->0, 9->1, 7->2, 2->3
    normalized_3 = {}
    for c, q in raw_map_3.items():
        if q in physical:
            normalized_3[c] = physical.index(q)
        elif 0 <= q < len(physical):
            normalized_3[c] = q
        else:
            raise ValueError(f"Invalid qubit {q}")
    # 输入: cbit 0=1 (binary 0001) -> 期望: logical bit 0=1 -> int 1
    counts_3 = {1: 10}

    def apply_measure_map(counts, measure_map):
        """将 counts 从 cbit 序转为逻辑序"""
        result = {}
        for ckey, count in counts.items():
            logical_idx = 0
            for j in range(4):
                cbit_j = (ckey >> j) & 1
                logic_bit = measure_map.get(j, j)
                if cbit_j:
                    logical_idx |= (1 << logic_bit)
            result[logical_idx] = result.get(logical_idx, 0) + count
        return result

    # Test case 1
    result_1 = apply_measure_map(counts_1, measure_map_1)
    expected_1 = {5: 10}
    assert result_1 == expected_1, f"Case1 failed: {result_1} != {expected_1}"
    print("Case1 (identity): PASS")

    # Test case 2
    result_2 = apply_measure_map(counts_2, measure_map_2)
    expected_2 = {2: 10}
    assert result_2 == expected_2, f"Case2 failed: {result_2} != {expected_2}"
    print("Case2 (swap 0<->1): PASS")

    # Test case 3: 物理编号场景
    result_3 = apply_measure_map(counts_3, normalized_3)
    expected_3 = {1: 10}  # cbit[0]=1 -> logical[0]=1 -> int 1
    assert result_3 == expected_3, f"Case3 failed: {result_3} != {expected_3}"
    print("Case3 (physical IDs): PASS")

    print("=== All tests passed ===\n")


def run_self_test():
    """运行自测"""
    test_measure_mapping()


if __name__ == "__main__":
    run_self_test()
