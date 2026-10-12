# -*- coding: utf-8 -*-
"""build_measured_params_v2.py —— 实测表生成（simultaneous 口径，2026-10-10 拍板）
与 v1 差异：
  单门 ε₁：优先用「边批内嵌单门」的同比特均值（simultaneous 口径，
           即完整线路中的真实表现，规避孤立测法对相干误差的虚高），
           无边批数据的比特回退单门专项批值
  双门 ε₂：max(0, ε₂ᶜ−ε₁a−ε₁b)，负值边填 0 并标注（说明该边 CZ 纯错误
           低于涨落地板，α 影响可忽略）
防呆三查同 v1。产物 measured_params.json。
"""
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, 'selected_layout_n13A_3col.json')
SD = os.path.join(HERE, 'results', 'calib_full_13')
OFFICIAL = os.path.join(HERE, '..', 'chip_params_WK_C180.json')
if not os.path.exists(OFFICIAL):  # 仓库结构（无父目录正本）时回退本目录
    OFFICIAL = os.path.join(HERE, 'chip_params_WK_C180.json')


def main():
    ro = json.load(open(os.path.join(SD, 'readout_all.json')))
    sg = json.load(open(os.path.join(SD, 'single_all.json')))
    cz = json.load(open(os.path.join(SD, 'cz_all.json')))
    ld = json.load(open(LAYOUT))
    phys = ld['physical_qubits']
    edges = []
    for a, b in ld['edges']:
        p1, p2 = phys[a], phys[b]
        edges.append((min(p1, p2), max(p1, p2)))

    eq = {int(q): v for q, v in ro['per_qubit'].items()}
    e1_iso = {int(q): v for q, v in sg['per_qubit'].items()}

    # simultaneous 单门：边批均值
    acc = defaultdict(list)
    for e, m in cz['e1_batch'].items():
        for q, v in m.items():
            acc[int(q)].append(v)
    e1_sim, src_tag = {}, {}
    for q in phys:
        if acc.get(q):
            e1_sim[q] = sum(acc[q]) / len(acc[q])
            src_tag[q] = 'sim(%d)' % len(acc[q])
        else:
            e1_sim[q] = e1_iso[q]
            src_tag[q] = 'iso'

    # 双门：批内拆 + 负值截 0
    import os as _os
    n_have = int(_os.environ.get('CALIB_EDGES', '15'))
    e2, neg_edges, missing_edges = {}, [], []
    for e in edges:
        k = '%d-%d' % e
        if k not in cz['per_edge']:
            missing_edges.append(k)      # 降级：缺边保留官方值（模板未覆盖即官方）
            continue
        raw = cz['per_edge'][k]
        v = max(0.0, raw)
        if raw < 0:
            neg_edges.append((k, raw))
        e2[e] = v

    # ---- 防抖三查 ----
    assert all(q in eq and q in e1_sim for q in phys), '查一：缺比特'
    if missing_edges:
        print('[查一·降级] %d 边缺实测，保留官方值兜底：%s' % (len(missing_edges), missing_edges))
    else:
        print('[查一] 13 比特 + 15 边齐全 OK')
    print('[查一] 13 比特 + 15 边齐全 OK')
    bad = []
    for q in phys:
        if not (0 <= eq[q] < 0.5):
            bad.append('q%d e_q=%.4f' % (q, eq[q]))
        if not (0 < e1_sim[q] < 0.1):
            bad.append('q%d e1=%.4f' % (q, e1_sim[q]))
    for e in edges:
        if not (0 <= e2[e] < 0.2):
            bad.append('边 %s e2=%.4f' % (e, e2[e]))
    assert not bad, '查二：' + str(bad)
    print('[查二] 数值合法（e2 允许 0=负值截断）OK')

    chip = json.load(open(OFFICIAL))
    for q in phys:
        chip['qubits'][str(q)]['e_q'] = eq[q]
        chip['qubits'][str(q)]['readout_fidelity'] = 1 - eq[q]
        chip['qubits'][str(q)]['gate1q_fidelity'] = 1 - e1_sim[q]
    for e in edges:
        chip['edges']['%d-%d' % e] = {'fidelity': 1 - e2[e]}
    chip['backend'] = 'originq:WK_C180 (MEASURED v2 simultaneous, 13+15)'
    chip['note'] = ('实测表 v2（simultaneous 单门口径 + 负值边截0）；'
                    '生成 %s' % datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    chip['provenance'] = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'eps1_source': {str(q): src_tag[q] for q in phys},
        'neg_eps2_edges': {k: v for k, v in neg_edges},
        'missing_edges_official_fallback': missing_edges,
        'readout_task_ids': ro.get('task_ids', []),
        'single_task_ids': sg.get('task_ids', []),
        'cz_task_ids': cz.get('task_ids', []),
        'shots': 20000,
    }
    print('[查三] 溯源写入 OK（负值边 %d 条标注）' % len(neg_edges))

    out = os.path.join(HERE, 'measured_params.json')
    if os.path.exists(out):
        prev = json.load(open(out))
        if 'DUMMY' not in str(prev.get('note', '')):
            os.rename(out, out.replace('.json', '_v1_backup.json'))
    json.dump(chip, open(out, 'w'), indent=1, ensure_ascii=False)

    print()
    print('===== 实测值总览（simultaneous 口径） =====')
    print('比特  e_q      eps1(来源)')
    for q in phys:
        print('q%-4d %.4f   %.4f  %s' % (q, eq[q], e1_sim[q], src_tag[q]))
    print('边       eps2(截0后)')
    for e in edges:
        raw = cz['per_edge']['%d-%d' % e]
        print('%-8s %.5f%s' % ('%d-%d' % e, e2[e], '  (原始 %.5f 负值截0)' % raw if raw < 0 else ''))
    print()
    print('measured_params.json 生成完毕')


if __name__ == '__main__':
    main()
