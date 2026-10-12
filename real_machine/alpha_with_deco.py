# -*- coding: utf-8 -*-
"""alpha_with_deco.py —— ④退相干项加回试算（2026-10-10，零机时）
口径：按⑬仲裁公式 F_idle(q) = (2·e^(−τ/T2*_q) + e^(−τ/T1_q)) / 3，
     α④ = α无④ × ∏_q F_idle(q)。
     τ 暂用线路总运行时长 T_total（全比特共用，改1 的"按总时长"方案；
     逐比特空闲时长需解析调度 IDLE 指令，此处近似先行）。
     T1/T2* 取实测表（模板继承官方逐比特值）。
产出：results/alpha_with_deco.json + 终端新表（不动已有双表）。
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
# API key：环境变量优先（仓库存户按 README 设置）；本机 ~/.originq_key 自动兜底
if not os.environ.get('ORIGINQ_API_KEY'):
    try:
        os.environ['ORIGINQ_API_KEY'] = open(os.path.expanduser('~/.originq_key')).read().strip()
    except OSError:
        pass

from uniqc import compile
from uniqc.backend_adapter import find_backend
import xeb_v2_real as XR

ld = XR.load_layout_from_json('selected_layout_n13A_3col.json', 0)
ca = ld['color_assignment']
phys = ld['physical_qubits']
chip = XR.load_chip_params('measured_params.json')
bi = find_backend('originq:WK_C180')

d = json.load(open('results/shallow_result.json'))
NS_PER_US = 1000.0

print('| m | F 实测 | α 无④ | α 含④ | F/α④ | α④/F | 偏离σ(④) | τ(ns) |')
print('|---|---|---|---|---|---|---|---|')
out = []
for r in d['rows']:
    m = r['m']
    # 代表线路的总时长（每档取首种子）
    circ = XR.generate_random_circuit(13, m, ca, seed=90000 + 100 * m)
    comp = compile(circ, backend_info=bi, available_qubits=phys)
    t_ns = XR.schedule_total_duration_ns(comp, bi)
    tau_us = t_ns / NS_PER_US
    # 逐比特退相干因子（T1/T2* 为 μs）
    deco = 1.0
    for q in phys:
        t1 = chip['qubits'][str(q)]['t1_us']
        t2 = chip['qubits'][str(q)]['t2star_us']
        f_idle = (2 * math.exp(-tau_us / t2) + math.exp(-tau_us / t1)) / 3
        deco *= f_idle
    a_no4 = r['alpha']
    a_4 = a_no4 * deco
    fm, se = r['F_mean'], r['F_SEM']
    ratio = fm / a_4 if a_4 > 1e-9 else None
    inv = (1 / ratio) if ratio and abs(ratio) > 1e-6 else None
    sig = (fm - a_4) / se if se > 0 else None
    print('| %d | %.4f | %.5f | %.6f | %s | %s | %s | %.0f |'
          % (m, fm, a_no4, a_4,
             ('%.2f' % ratio) if ratio is not None else '—',
             ('%.2f' % inv) if inv is not None else '—',
             ('%+.1f' % sig) if sig is not None else '—',
             t_ns))
    out.append({'m': m, 'F': fm, 'alpha_no4': a_no4, 'alpha_with4': a_4,
                'deco_factor': deco, 't_total_ns': t_ns,
                'ratio': ratio, 'sigma': sig})

json.dump(out, open('results/alpha_with_deco.json', 'w'), indent=1)
print('saved results/alpha_with_deco.json')
