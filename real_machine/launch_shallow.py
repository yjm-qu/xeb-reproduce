# -*- coding: utf-8 -*-
"""launch_shallow.py —— 浅档验证批发射器（m=9,12,15,18 × k=30，2026-10-10）
断网免疫模式：只提交不等待，task_id+线路元数据落盘 pending_shallow.json。
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

from uniqc import compile, submit_batch
from uniqc.backend_adapter import find_backend
import xeb_v2_real as XR

BACKEND = 'originq:WK_C180'
SHOTS = 20000
MS = [9, 12, 15, 18]
K = 30
SEED_BASE = 90000

bi = find_backend(BACKEND)
ld = XR.load_layout_from_json('selected_layout_n13A_3col.json', 0)
ca = ld['color_assignment']
phys = ld['physical_qubits']
chip = XR.load_chip_params('measured_params.json')

jobs = []
for m in MS:
    cc = {}
    for i in range(m):
        cc[XR.PATTERN_CYCLE[i % 6]] = cc.get(XR.PATTERN_CYCLE[i % 6], 0) + 1
    exp_cz = sum(len(ca[c]) * cc.get(c, 0) for c in ca)
    alpha = XR.compute_alpha_f_exact(ld, chip, 13, m, ca)[0]
    for kk in range(K):
        seed = SEED_BASE + 100 * m + kk
        circ = XR.generate_random_circuit(13, m, ca, seed=seed)
        comp = compile(circ, backend_info=bi, available_qubits=phys)
        ncz = sum(1 for l in comp.originir.split('\n') if l.strip().startswith('CZ'))
        assert ncz == exp_cz, 'seed %s CZ=%d != %d' % (seed, ncz, exp_cz)
        psi = XR.Simulator('statevector').simulate_statevector(circ)
        norm = {}
        for line in comp.originir.split('\n'):
            line = line.strip()
            if line.startswith('MEASURE'):
                parts = line[7:].strip().split(',')
                q = int(parts[0].split('[')[1].split(']')[0])
                cb = int(parts[1].split('[')[1].split(']')[0])
                norm[cb] = phys.index(q) if q in phys else q
        jobs.append({'m': m, 'seed': seed, 'norm': {str(k): v for k, v in norm.items()},
                     'psi_re': [float(x.real) for x in psi],
                     'psi_im': [float(x.imag) for x in psi],
                     'compiled': comp})
    print('m=%2d: %d 条构建（CZ=%d，α=%.5f）' % (m, K, exp_cz, alpha), flush=True)

tid = submit_batch([j['compiled'] for j in jobs], backend=BACKEND,
                   shots=SHOTS, local_compile=0, cloud_compile=0)
print('[浅档批] task=%s 已上云（%d 条）' % (tid, len(jobs)), flush=True)

def ser(j):
    return {'m': j['m'], 'seed': j['seed'], 'norm': j['norm'],
            'psi': {'re': j['psi_re'], 'im': j['psi_im']}}
json.dump({'launched_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'shots': SHOTS, 'task_id': tid, 'n_circuits': len(jobs),
           'ms': MS, 'k': K, 'seed_base': SEED_BASE,
           'jobs': [ser(j) for j in jobs]},
          open('pending_shallow.json', 'w'), indent=1, ensure_ascii=False)
print('任务档案: pending_shallow.json')
