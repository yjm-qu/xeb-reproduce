# -*- coding: utf-8 -*-
"""WK_C180 真机标定实验 v3.1（2026-09-24，差分边标定 + 落点修正 + QINIT 修正）
平台限制: 无 CZ 的线路编译时退化放置(无视 available_qubits)。
对策: 编译后核验实际比特使用, 退化则文本改号到目标比特(单比特门无连接约束, 合法)。
三层:
  L1 读出: 2 任务 (全|0> / 全|1>), 13 比特同测
  L2 单门: 15 边 × X 类(无CZ, 纯随机单门) × m十档 × 5种子 -> 每边单门对贡献
  L3 双门: 15 边 × Y 类(同序列+CZ) × m十档 × 5种子 -> CZ贡献 = Y衰减 - X衰减
用法: python3 calibrate_real.py --dry-run  |  python3 calibrate_real.py
"""
import os, sys, json, random, argparse, re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

# API key：环境变量优先（仓库存户按 README 设置）；本机 ~/.originq_key 自动兜底
if not os.environ.get('ORIGINQ_API_KEY'):
    try:
        os.environ['ORIGINQ_API_KEY'] = open(os.path.expanduser('~/.originq_key')).read().strip()
    except OSError:
        pass

import numpy as np
import xeb_v2_real as xeb
from uniqc import Circuit, submit_batch, wait_for_result, compile
from uniqc.backend_adapter import find_backend
from uniqc.simulator import Simulator

BACKEND = 'originq:WK_C180'
N = 13
SG = ['A', 'B', 'C']
PHI = {'A': 0.0, 'B': 1.5707963267948966, 'C': 0.7853981633974483}
TH = 1.5707963267948966
M2 = [2, 5, 10, 20, 50, 100, 200, 500, 750, 1000]
SEEDS = [0, 1, 2, 3, 4]
SHOTS = 2000
LAYOUT_FILE = 'selected_layout_n13A_3col.json'
CHIP_FILE = '../chip_params_WK_C180.json'

xeb.BACKEND = BACKEND
xeb.CHIP_PARAMS_JSON_PATH = CHIP_FILE

TASKS = []
_QRE = re.compile(r'q\[(\d+)\]')


def parse_norm_used(comp):
    """返回 (norm{cbit->物理q}, 使用的物理比特集合)"""
    norm, used = {}, set()
    for line in comp.originir.split('\n'):
        line = line.strip()
        for m in _QRE.finditer(line):
            used.add(int(m.group(1)))
        if line.startswith('MEASURE'):
            parts = line[7:].strip().split(',')
            q = int(parts[0].split('[')[1].split(']')[0])
            cb = int(parts[1].split('[')[1].split(']')[0])
            norm[cb] = q
    return norm, used


def fix_placement(comp, target):
    """核验编译放置; 退化则文本改号到目标比特。target: {cbit: 物理比特}。"""
    norm, used = parse_norm_used(comp)
    if set(used) == set(target.values()):
        return comp, norm
    mapping = {q: target[cb] for cb, q in norm.items()}
    lines = []
    for line in comp.originir.split('\n'):
        if line.strip().startswith('QINIT'):
            line = 'QINIT %d' % (max(mapping.values()) + 1)
        else:
            line = _QRE.sub(lambda m: 'q[%d]' % mapping.get(int(m.group(1)),
                                                            int(m.group(1))), line)
        lines.append(line)
    new_o = '\n'.join(lines)
    new_comp = Circuit.from_originir(new_o)
    norm2, used2 = parse_norm_used(new_comp)
    assert set(used2) == set(target.values()), '改号后仍越界: %s' % sorted(used2)
    return new_comp, norm2


def random_pair_sequence(m, rng):
    s0, s1 = [], []
    l0 = l1 = None
    for _ in range(m):
        g0 = rng.choice([g for g in SG if g != l0])
        l0 = g0
        g1 = rng.choice([g for g in SG if g != l1])
        l1 = g1
        s0.append(g0)
        s1.append(g1)
    return s0, s1


def build_edge_circuit(seq0, seq1, with_cz):
    c = Circuit(2)
    for cyc in range(len(seq0)):
        c.rphi(0, theta=TH, phi=PHI[seq0[cyc]])
        c.rphi(1, theta=TH, phi=PHI[seq1[cyc]])
        if with_cz:
            c.cz(0, 1)
    c.measure(0)
    c.measure(1)
    return c


def build_all(bi, ld):
    phys = ld['physical_qubits']
    edges = sorted({tuple(sorted((phys[e[0]], phys[e[1]]))) for e in ld['edges']})

    # ---- L1 读出: 2 任务, 全 13 比特 (退化放置 -> 改号回簇) ----
    for st, name in [(0, 'g0'), (1, 'g1')]:
        c = Circuit(N)
        if st == 1:
            for qi in range(N):
                c.rphi(qi, theta=3.141592653589793, phi=0.0)
        for qi in range(N):
            c.measure(qi)
        comp = compile(c, backend_info=bi, available_qubits=phys)
        target = {cb: phys[cb] for cb in range(N)}
        comp, norm = fix_placement(comp, target)
        TASKS.append({'kind': 'L1', 'tag': 'readout_%s' % name, 'prep': st,
                      'meta': {'used': sorted(norm.values())}, 'compiled': comp,
                      'psi': None, 'norm': norm})

    # ---- L2/L3: 逐边差分 (X 类退化 -> 改号到边端点; Y 类编译即正确) ----
    for e_idx, (a, b) in enumerate(edges):
        for m in M2:
            for k in SEEDS:
                seed = 42000 + e_idx * 1000 + m * 10 + k
                rng = random.Random(seed)
                seq0, seq1 = random_pair_sequence(m, rng)
                for kind in ('X', 'Y'):
                    c = build_edge_circuit(seq0, seq1, with_cz=(kind == 'Y'))
                    comp = compile(c, backend_info=bi, available_qubits=[a, b])
                    target = {0: a, 1: b}
                    comp, norm = fix_placement(comp, target)
                    psi = np.array(Simulator('statevector').simulate_statevector(c))
                    psi = psi / np.linalg.norm(psi)
                    TASKS.append({'kind': 'L2' if kind == 'X' else 'L3',
                                  'tag': 'e%d-%d_%s_m%d_s%d' % (a, b, kind, m, k),
                                  'meta': {'edge': [a, b], 'kind': kind, 'm': m,
                                           'seed': seed},
                                  'compiled': comp, 'psi': psi, 'norm': norm})
    cnt = {k: sum(1 for t in TASKS if t['kind'] == k) for k in ('L1', 'L2', 'L3')}
    print('L1:%d  L2(X类):%d  L3(Y类):%d  total:%d'
          % (cnt['L1'], cnt['L2'], cnt['L3'], len(TASKS)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    ld = xeb.load_layout_from_json(LAYOUT_FILE, N)
    ld['backend'] = BACKEND
    json.dump(ld, open(LAYOUT_FILE, 'w'), indent=1)
    chip = xeb.load_chip_params(CHIP_FILE)
    xeb.check_backend_alignment(ld, BACKEND)
    xeb.check_topology_alignment(ld, BACKEND)
    xeb.check_data_holes(ld, chip)

    bi = find_backend(BACKEND)
    build_all(bi, ld)

    # 自检: 每个任务的实际比特使用必须与登记一致
    bad = []
    for t in TASKS:
        norm, used = parse_norm_used(t['compiled'])
        if set(used) != set(norm.values()):
            bad.append(t['tag'])
    print('[自检] 全部 %d 任务比特使用核验: 异常 %d 个' % (len(TASKS), len(bad)),
          flush=True)
    assert not bad, bad[:5]
    if args.dry_run:
        print('[DRY-RUN] checks passed, NOT submitting.', flush=True)
        return

    task_id = submit_batch([t['compiled'] for t in TASKS], backend=BACKEND, shots=SHOTS)
    print('batch task:', task_id, flush=True)
    json.dump({'task_id': task_id, 'shots': SHOTS,
               'tasks': [{k: t[k] for k in ('kind', 'tag', 'meta', 'norm')} for t in TASKS]},
              open('calib_jobs.json', 'w'), indent=1)
    print('waiting (timeout 10800s)...', flush=True)
    res = wait_for_result(task_id, timeout=10800)
    print('results received', flush=True)
    items = res if isinstance(res, list) else (res.results if hasattr(res, 'results') else res.get('results'))
    assert len(items) == len(TASKS), 'count mismatch %d vs %d' % (len(items), len(TASKS))
    json.dump({'task_id': task_id,
               'results': [{'tag': t['tag'], 'kind': t['kind'], 'meta': t['meta'],
                            'norm': t['norm'],
                            'counts': (it.counts if hasattr(it, 'counts') else it.get('counts'))}
                           for t, it in zip(TASKS, items)]},
              open('calib_raw.json', 'w'), indent=1)
    print('saved calib_raw.json (分析: python3 calib_analyze.py)', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
