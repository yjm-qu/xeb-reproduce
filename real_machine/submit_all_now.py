# -*- coding: utf-8 -*-
"""submit_all_now.py —— 断网免疫发射器（2026-10-10）
一次性把「剩余标定边（合并一批）+ 主实验 30 条」全部推上云端，
只提交不等待，任务号落盘 pending_cloud.json 后立即退出。
断网后云端照跑；回来跑 fetch_all.py 取结果。

主实验线路不依赖标定表（α 是事后本地算的），构建用官方表参考。
"""
import json
import os
import sys
import time

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

from uniqc import Circuit, submit_batch, compile
from uniqc.backend_adapter import find_backend
import calib_full_13 as CF
import xeb_v2_real as XR

BACKEND = 'originq:WK_C180'
SHOTS = 20000
REMAIN_EDGES = [(122, 131), (123, 132), (131, 141), (132, 141)]
MAIN_K, MAIN_M, MAIN_SEED_BASE = 30, 21, 90000

bi = find_backend(BACKEND)
pending = {'launched_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'shots': SHOTS}

# ============ 1. 剩余标定边合并一批（4 边 × 240 条） ============
tasks = []
for edge in REMAIN_EDGES:
    a, b = edge
    tasks += CF.build_cz_tasks(bi, edge)      # CZ 80
    tasks += CF.build_1q_tasks(bi, a, b)      # 端 a 单门 80
    tasks += CF.build_1q_tasks(bi, b, a)      # 端 b 单门 80
print('[标定合并批] %d 边 × 240 = %d 条构建完成（预检全过）'
      % (len(REMAIN_EDGES), len(tasks)), flush=True)
# 构建元数据随任务号存盘（fetch 时按此切分打分）
meta = [{'kind': t['kind'], 'edge_order': i // 240,
         'm': t['m'], 'k': t['k'],
         'q': t.get('q'), 'cbit0': t.get('cbit0'), 'cbit1': t.get('cbit1'),
         'p_exp': list(np.asarray(t['p_exp'], float))}
        for i, t in enumerate(tasks)]
tid = submit_batch([t['compiled'] for t in tasks], backend=BACKEND,
                   shots=SHOTS, local_compile=0, cloud_compile=0)
print('[标定合并批] task=%s 已上云' % tid, flush=True)
pending['calib_merge'] = {'task_id': tid, 'edges': [list(e) for e in REMAIN_EDGES],
                          'n_circuits': len(tasks), 'meta': meta}

# ============ 2. 主实验 30 条（线路与标定表无关） ============
ld = XR.load_layout_from_json('selected_layout_n13A_3col.json', 0)
ca = ld['color_assignment']
phys = ld['physical_qubits']
chip = XR.load_chip_params('../chip_params_WK_C180.json')   # 参考表（α 事后重算）
jobs = []
for kk in range(MAIN_K):
    seed = MAIN_SEED_BASE + 100 * MAIN_M + kk
    circ = XR.generate_random_circuit(13, MAIN_M, ca, seed=seed)
    comp = compile(circ, backend_info=bi, available_qubits=phys)
    ncz = sum(1 for l in comp.originir.split('\n') if l.strip().startswith('CZ'))
    assert ncz == 105, 'seed %s CZ=%d' % (seed, ncz)
    psi = XR.Simulator('statevector').simulate_statevector(circ)
    norm = {}
    for line in comp.originir.split('\n'):
        line = line.strip()
        if line.startswith('MEASURE'):
            parts = line[7:].strip().split(',')
            q = int(parts[0].split('[')[1].split(']')[0])
            cb = int(parts[1].split('[')[1].split(']')[0])
            norm[cb] = phys.index(q) if q in phys else q
    jobs.append({'seed': seed, 'psi': list(np.asarray(psi, complex)),
                 'norm': {str(k): v for k, v in norm.items()},
                 'compiled': comp})
print('[主实验批] %d 条构建完成（CZ=105 逐条自检过）' % len(jobs), flush=True)
tid2 = submit_batch([j['compiled'] for j in jobs], backend=BACKEND,
                    shots=SHOTS, local_compile=0, cloud_compile=0)
print('[主实验批] task=%s 已上云' % tid2, flush=True)
# psi/norm 是 numpy 复数，存档转可序列化格式
def ser_psi(p):
    return {'re': [float(x.real) for x in p], 'im': [float(x.imag) for x in p]}
pending['main_exp'] = {
    'task_id': tid2, 'n_circuits': len(jobs), 'm': MAIN_M,
    'k': MAIN_K, 'seed_base': MAIN_SEED_BASE,
    'jobs': [{'seed': j['seed'], 'norm': j['norm'], 'psi': ser_psi(j['psi'])}
             for j in jobs]}

json.dump(pending, open('pending_cloud.json', 'w'), indent=1, ensure_ascii=False)
print()
print('===== 全部上云，断网免疫完成 =====')
print('标定合并批: %s（%d 条）' % (tid, len(tasks)))
print('主实验批  : %s（%d 条）' % (tid2, len(jobs)))
print('任务档案  : pending_cloud.json')
print('回来联网后跑: python3 fetch_all.py')
