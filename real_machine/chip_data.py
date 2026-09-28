#!/usr/bin/env python3
"""
XLSX → JSON 解析器：解析本源芯片标定数据
不涉及量子操作，不 import uniqc
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# 依赖已安装: openpyxl, pandas
import pandas as pd


def parse_cz_column(cz_str):
    """
    解析 CZ 列字符串
    格式: "CZ39_48:0.9901;CZ39_49:0.995" 或 "CZ39_48:0.9901;"
    返回: dict {(q1, q2): fidelity}
    """
    if pd.isna(cz_str) or not cz_str.strip():
        return {}

    cz_str = str(cz_str).strip()
    if cz_str.endswith(';'):
        cz_str = cz_str[:-1]

    result = {}
    for item in cz_str.split(';'):
        item = item.strip()
        if not item:
            continue
        # 格式: CZ39_48:0.9901
        if ':' not in item:
            continue
        name_part, fid_part = item.rsplit(':', 1)
        # 名字形如 CZ39_48
        if not name_part.startswith('CZ'):
            continue
        numbers = name_part[2:]  # 39_48
        q1_str, q2_str = numbers.split('_')
        q1, q2 = int(q1_str), int(q2_str)
        fidelity = float(fid_part)
        key = (min(q1, q2), max(q1, q2))
        result[key] = fidelity

    return result


def parse_xlsx(xlsx_path, backend_name):
    """解析 XLSX 文件"""
    print(f"Reading: {xlsx_path}")

    # 读取唯一的 sheet 'qubit'
    df = pd.read_excel(xlsx_path, sheet_name='qubit')
    print(f"Sheet 'qubit': {len(df)} rows")

    # 打印表头
    print(f"\nColumns: {list(df.columns)}")

    # 解析每个比特
    qubits = {}
    all_edges = {}  # {(q1, q2): fidelity}

    for idx, row in df.iterrows():
        # 解析比特编号: q39 → 39
        qubit_name = str(row['Qubits']).strip()
        if not qubit_name.startswith('q'):
            continue
        qubit_id = int(qubit_name[1:])

        # 读取各项参数
        freq_mhz = float(row['Frequency(MHz)'])
        t1_us = float(row['T1(μs)'])
        t2star_us = float(row['T2*(μs)'])
        t2echo_us = float(row['T2,Echo(μs)'])
        readout_fid = float(row['Readout Fidelity'])
        f0 = float(row['F0 Readout Fidelity'])
        f1 = float(row['F1 Readout Fidelity'])
        gate1q_fid = float(row['1-Qubit Gate Fidelity'])

        # e_q 计算
        eq = 1 - readout_fid
        eq_check = ((1 - f0) + (1 - f1)) / 2

        # 交叉校验
        if abs(eq - eq_check) > 1e-6:
            raise ValueError(
                f"e_q cross-check failed for q{qubit_id}: "
                f"eq={eq:.8f}, eq_check={eq_check:.8f}, diff={abs(eq-eq_check):.10f}"
            )

        qubits[str(qubit_id)] = {
            "e_q": eq,
            "readout_fidelity": readout_fid,
            "f0": f0,
            "f1": f1,
            "gate1q_fidelity": gate1q_fid,
            "t1_us": t1_us,
            "t2star_us": t2star_us,
            "t2echo_us": t2echo_us,
            "freq_mhz": freq_mhz,
        }

        # 解析 CZ 边
        cz_str = row['CZ Gate Fidelity']
        edges = parse_cz_column(cz_str)

        for (q1, q2), fid in edges.items():
            key = (min(q1, q2), max(q1, q2))
            key_str = f"{q1}-{q2}"

            if key_str in all_edges:
                old_fid = all_edges[key_str]["fidelity"]
                if old_fid != fid:
                    # 取小数位更多的
                    old_str = f"{old_fid}"
                    new_str = f"{fid}"
                    if len(new_str) > len(old_str):
                        print(f"  WARNING: Edge {key_str} appears with different fidelity: "
                              f"{old_fid} vs {fid}, keeping {fid}")
                        all_edges[key_str] = {"fidelity": fid}
                    else:
                        print(f"  WARNING: Edge {key_str} appears with different fidelity: "
                              f"{old_fid} vs {fid}, keeping {old_fid}")
            else:
                all_edges[key_str] = {"fidelity": fid}

    # 汇总统计
    eq_values = [q["e_q"] for q in qubits.values()]
    summary = {
        "n_qubits": len(qubits),
        "n_edges": len(all_edges),
        "avg_eq": sum(eq_values) / len(eq_values),
        "min_eq": min(eq_values),
        "max_eq": max(eq_values),
    }

    # 打印解析结果
    print(f"\n{'='*60}")
    print(f"Parsed {len(qubits)} qubits, {len(all_edges)} edges")
    print(f"{'='*60}")

    print("\n--- Qubits ---")
    for qid in sorted(qubits.keys(), key=int):
        q = qubits[qid]
        print(f"q{qid}: e_q={q['e_q']:.4f}, 1q_fid={q['gate1q_fidelity']:.4f}")

    print("\n--- Edges ---")
    for edge in sorted(all_edges.keys(), key=lambda x: list(map(int, x.split('-')))):
        print(f"Edge {edge}: fidelity={all_edges[edge]['fidelity']:.4f}")

    print(f"\n--- Summary ---")
    print(f"n_qubits: {summary['n_qubits']}")
    print(f"n_edges: {summary['n_edges']}")
    print(f"avg_eq: {summary['avg_eq']:.4f}")
    print(f"min_eq: {summary['min_eq']:.4f}")
    print(f"max_eq: {summary['max_eq']:.4f}")

    # 构建输出 JSON
    result = {
        "backend": backend_name,
        "source_file": str(Path(xlsx_path).resolve()),
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "qubits": qubits,
        "edges": all_edges,
        "summary": summary,
    }

    return result


def parse_rows(rows_str):
    """解析 rows 字符串: "39,40;48,49,50;57,58,59;66,67,68;75,76,77;84,85,86"
    返回: List[List[int]], positions dict
    """
    rows = []
    all_qubits = set()
    for row_str in rows_str.split(';'):
        row = [int(q.strip()) for q in row_str.split(',') if q.strip()]
        rows.append(row)
        all_qubits.update(row)

    # 构建 qubit -> (row, col) 映射
    positions = {}
    for r, row in enumerate(rows):
        for c, q in enumerate(row):
            positions[q] = (r, c)

    return rows, positions


def validate_rows_edges(rows, all_edges):
    """校验所有边必须连接相邻两行"""
    qubit_to_row = {}
    for r, row in enumerate(rows):
        for q in row:
            qubit_to_row[q] = r

    for edge_str in all_edges.keys():
        q1, q2 = map(int, edge_str.split('-'))
        r1 = qubit_to_row.get(q1)
        r2 = qubit_to_row.get(q2)

        if r1 is None or r2 is None:
            continue  # 跳过不在 rows 中的边

        row_diff = abs(r2 - r1)
        assert row_diff == 1, \
            f"Edge {edge_str} connects row {r1} and {r2} (diff={row_diff}), must be adjacent (diff=1)"


def main():
    parser = argparse.ArgumentParser(description='Parse OriginQ chip XLSX to JSON')
    parser.add_argument('--xlsx', required=True, help='Input XLSX file path')
    parser.add_argument('--backend', required=True, help='Backend name, e.g. originq:WK_C180_2')
    parser.add_argument('--out', required=True, help='Output JSON file path')
    parser.add_argument('--rows', help='Chip rows, e.g. "39,40;48,49,50;..."')
    args = parser.parse_args()

    # 解析
    result = parse_xlsx(args.xlsx, args.backend)

    # 如果提供了 rows，校验并写入
    if args.rows:
        rows, positions = parse_rows(args.rows)
        result['rows'] = rows
        result['positions'] = positions

        # 校验边连接相邻行
        validate_rows_edges(rows, result['edges'])
        print(f"\nRows validation passed: all {len(result['edges'])} edges connect adjacent rows")

    # 硬性校验
    assert result["summary"]["n_qubits"] == 17, \
        f"Expected 17 qubits, got {result['summary']['n_qubits']}"
    assert result["summary"]["n_edges"] == 20, \
        f"Expected 20 edges, got {result['summary']['n_edges']}"

    # 验证 JSON 可重新加载
    json_str = json.dumps(result, indent=2)
    loaded = json.loads(json_str)
    assert loaded is not None, "JSON reload verification failed"

    # 写入输出
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Output written to: {args.out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
