# -*- coding: utf-8 -*-
"""fetch_all.py —— 取回云端两批结果（2026-10-10 断网免疫方案配套）
用法：
  python3 fetch_all.py main       # 取主实验批：算 F（psi/norm 从 pending_cloud.json）
  python3 fetch_all.py calib      # 取标定合并批：4 边打分拟合，并入 cz_all.json
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

from uniqc import wait_for_result
import calib_full_13 as CF

PEND = json.load(open('pending_cloud.json'))


def get_items(tid):
    res = wait_for_result(tid, timeout=60)
    items = res if isinstance(res, list) else \
        (res.results if hasattr(res, 'results') else res.get('results'))
    return [(it.counts if hasattr(it, 'counts') else it.get('counts'))
            for it in items]


def fetch_main():
    d = PEND['main_exp']
    print('[主实验] task=%s 取回中...' % d['task_id'], flush=True)
    counts_list = get_items(d['task_id'])
    assert len(counts_list) == d['n_circuits'], \
        '条数 %d != %d' % (len(counts_list), d['n_circuits'])
    print('[主实验] %d/%d 条取回' % (len(counts_list), d['n_circuits']), flush=True)
    import xeb_v2_real as XR
    rows = []
    for j, counts in zip(d['jobs'], counts_list):
        psi = np.array([complex(r, i) for r, i in zip(j['psi']['re'], j['psi']['im'])])
        norm = {int(k): v for k, v in j['norm'].items()}
        n = 13
        bits = []
        for kk_, cnt in counts.items():
            s = str(kk_).strip()
            if s[:2].lower() == '0b':
                s = s[2:]
            cidx = int(s, 2) if any(c in '01' for c in s) else int(s)
            logical = 0
            for b in range(n):
                if (cidx >> b) & 1:
                    logical |= (1 << norm.get(b, b))
            bits.extend([logical] * cnt)
        F = XR.compute_F_XEB(psi, bits, n)
        rows.append({'seed': j['seed'], 'F': round(float(F), 4)})
        print('  seed=%s F=%.4f' % (j['seed'], F), flush=True)
    fs = np.array([r['F'] for r in rows])
    out = {'task_id': d['task_id'], 'm': d['m'], 'k': d['k'], 'shots': PEND['shots'],
           'per_seed': rows, 'F_mean': round(float(fs.mean()), 4),
           'F_SEM': round(float(fs.std(ddof=1) / np.sqrt(len(fs))), 4)}
    json.dump(out, open('results/main_exp_result.json', 'w'), indent=1)
    print('===== 主实验 F_mean=%.4f ± %.4f =====' % (out['F_mean'], out['F_SEM']))
    print('saved results/main_exp_result.json')
    print('（α 待标定合并批返回后用实测表计算）')


def fetch_calib():
    d = PEND['calib_merge']
    print('[标定合并批] task=%s 取回中...' % d['task_id'], flush=True)
    counts_list = get_items(d['task_id'])
    assert len(counts_list) == d['n_circuits'], \
        '条数 %d != %d' % (len(counts_list), d['n_circuits'])
    print('[标定合并批] %d 条取回' % len(counts_list), flush=True)
    # 读取已有 cz_all（11 边）增量并入
    cz = json.load(open('results/calib_full_13/cz_all.json'))
    for e_idx, edge in enumerate(d['edges']):
        a, b = edge
        lo, hi = e_idx * 240, (e_idx + 1) * 240
        cz_rows, q_rows = {}, {}
        for m_, counts in zip(d['meta'][lo:hi], counts_list[lo:hi]):
            p_exp = np.array(m_['p_exp'])
            if m_['kind'] == 'cz':
                dist = CF.counts_to_dist(counts, 2, m_['cbit0'], m_['cbit1'])
                f = CF.fxeb_neill(dist, p_exp, CF.D4)
                if f is not None:
                    cz_rows.setdefault(m_['m'], []).append(f)
            else:
                dist = CF.counts_to_dist(counts, 1, 0)
                f = CF.counts_to_dist(counts, 1, 0)
                f = CF.fxeb_neill(dist, p_exp, CF.D1)
                if f is not None:
                    q_rows.setdefault(m_['q'], {}).setdefault(m_['m'], []).append(f)
        ms_cz = sorted(cz_rows)
        means_cz = [float(np.mean(cz_rows[m])) for m in ms_cz]
        assert means_cz[0] - means_cz[-1] >= CF.DECAY_GUARD, \
            '边 %s CZ 无衰减 %s' % (edge, np.round(means_cz, 3))
        eps2c = CF.fit_eps(ms_cz, means_cz, CF.D4)
        e1_local = {}
        for q in (a, b):
            per_m = q_rows[q]
            ms = sorted(per_m)
            means = [float(np.mean(per_m[mm])) for mm in ms]
            e1_local[q] = CF.fit_eps(ms, means, CF.D1)
        eps2 = max(0.0, eps2c - e1_local[a] - e1_local[b])
        k = '%d-%d' % (a, b)
        cz['per_edge'][k] = eps2
        cz['eps2c'][k] = eps2c
        cz['e1_batch'][k] = {str(a): e1_local[a], str(b): e1_local[b]}
        print('  边 %d-%d eps2c=%.5f eps2=%.5f%s'
              % (a, b, eps2c, eps2,
                 '（负值截0）' if eps2c - e1_local[a] - e1_local[b] < 0 else ''), flush=True)
    json.dump(cz, open('results/calib_full_13/cz_all.json', 'w'), indent=1)
    print('===== 双门 %d/15 边齐 =====' % len(cz['per_edge']))
    print('saved results/calib_full_13/cz_all.json')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('main', 'all'):
        fetch_main()
    if which in ('calib', 'all'):
        fetch_calib()
