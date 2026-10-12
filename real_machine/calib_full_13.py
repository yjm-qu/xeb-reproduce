# -*- coding: utf-8 -*-
"""calib_full_13.py —— 三项错误率标定铺开（13 比特 + 15 边，2026-10-08 章程第 1 步）
（v2 整文件重写：v1 多轮补丁后 phase_single/phase_cz 结构损坏）

复用三套已验收的 uniqc 版标定逻辑（compile(level=0)+双开关，SHOTS=20000）：
  阶段① 读出：13 比特 × 2 线路（|0> 直测 / X 后测），1 批 26 条
  阶段② 单门：13 比特 × 8 档 × 10 串，逐比特 1 批（80 条）
  阶段③ 双门：15 边 ×（CZ 80 + 两端单门 80×2 = 240 条），逐边 1 批
逐批：条数强校验 → 打分拟合 → 增量落盘 results/calib_full_13/。
任一批失败立即停（已完成批次保留有效，支持 --phase 断点续跑）。
产出：readout_all.json / single_all.json / cz_all.json（供 build_measured_params.py）

用法：
  python3 calib_full_13.py --dry-run                # 本地构建自检，不提交
  python3 calib_full_13.py --submit                 # 真机全量（①→②→③）
  python3 calib_full_13.py --submit --phase single  # 断点续跑某阶段
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# API key：环境变量优先（仓库存户按 README 设置）；本机 ~/.originq_key 自动兜底
if not os.environ.get('ORIGINQ_API_KEY'):
    try:
        os.environ['ORIGINQ_API_KEY'] = open(os.path.expanduser('~/.originq_key')).read().strip()
    except OSError:
        pass

from uniqc import Circuit, submit_batch, wait_for_result, compile
from uniqc.backend_adapter import find_backend
from calibrate_real import fix_placement

BACKEND = 'originq:WK_C180'
SHOTS = 20000
LAYOUT = 'selected_layout_n13A_3col.json'
OUTDIR = os.path.join('results', 'calib_full_13')
D1, D4 = 2, 4
THETA = math.pi / 2
PHI = {'sqrtX': 0.0, 'sqrtY': math.pi / 2, 'sqrtW': math.pi / 4}
M_VALUES = [2, 5, 10, 20, 50, 100, 200, 500]
N_SEQ = 10
DECAY_GUARD = 0.05
MASTER_SEED = 20261008


# ============================================================
# 公共数学件（与已验收标定脚本同款）
# ============================================================
def rphi_matrix(theta, phi):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * np.exp(-1j * phi) * s],
                     [-1j * np.exp(1j * phi) * s, c]])


def draw_seq(m, rng):
    seq, prev = [], None
    for _ in range(m):
        g = rng.choice([g for g in PHI if g != prev])
        prev = g
        seq.append((THETA, PHI[g], g))
    return seq


def ideal_1q(seq):
    v = np.array([1, 0], dtype=complex)
    for theta, phi, _g in seq:
        v = rphi_matrix(theta, phi) @ v
        v = v / np.linalg.norm(v)
    return np.abs(v) ** 2


def ideal_2q(seq0, seq1):
    v = np.array([1, 0, 0, 0], dtype=complex)
    for k in range(len(seq0)):
        v2 = v.reshape(2, 2)
        v2 = rphi_matrix(*seq0[k][:2]) @ v2
        v2 = v2 @ rphi_matrix(*seq1[k][:2]).T
        v2[1, 1] *= -1
        v = v2.flatten()
        v = v / np.linalg.norm(v)
    return np.abs(v) ** 2


def is_degenerate(p_exp):
    p_exp = np.asarray(p_exp, float)
    p_in = np.full(len(p_exp), 1.0 / len(p_exp))
    m = p_exp > 0
    den = float(-np.sum(p_in[m] * np.log2(p_exp[m])) +
                np.sum(p_exp[m] * np.log2(p_exp[m])))
    return den < 1e-12


def self_entropy(p):
    p = np.asarray(p, float)
    m = p > 0
    return float(-np.sum(p[m] * np.log2(p[m])))


def cross_entropy(p, qv):
    p, qv = np.asarray(p, float), np.asarray(qv, float)
    m = (p > 0) & (qv > 0)
    return float(-np.sum(p[m] * np.log2(qv[m])))


def fxeb_neill(p_meas, p_exp, d):
    p_in = np.full(len(p_exp), 1.0 / d)
    num = cross_entropy(p_in, p_exp) - cross_entropy(p_meas, p_exp)
    den = cross_entropy(p_in, p_exp) - self_entropy(p_exp)
    if abs(den) < 1e-12:
        return None
    return num / den


def fit_eps(ms, fs, d):
    model = lambda m, eps: (1 - eps / (1 - 1.0 / d ** 2)) ** m
    return float(curve_fit(model, np.array(ms, float), np.array(fs, float),
                           p0=[0.01], bounds=(0, 1), maxfev=10000)[0])


def counts_to_dist(counts, n_bits, cbit0, cbit1=None):
    """uniqc counts -> 概率分布；1 比特 idx=bit(cbit0)，2 比特 idx=q0+2*q1"""
    out = {}
    for k, c in counts.items():
        s = str(k).strip()
        if s[:2].lower() == '0b':
            v = int(s[2:], 2)
        else:
            try:
                v = int(s, 2)
            except ValueError:
                v = int(s)
        if n_bits == 1:
            idx = (v >> cbit0) & 1
        else:
            idx = ((v >> cbit0) & 1) + 2 * ((v >> cbit1) & 1)
        out[idx] = out.get(idx, 0) + c
    tot = sum(out.values())
    return np.array([out.get(i, 0) / tot for i in range(2 ** n_bits)])


def parse_compiled(comp):
    """MEASURE 行 -> {物理号: cbit}；并数 SX/CZ 脉冲"""
    meas, n_sx, n_cz = {}, 0, 0
    for line in comp.originir.split('\n'):
        line = line.strip()
        if line.startswith('MEASURE'):
            parts = line[7:].strip().split(',')
            q = int(parts[0].split('[')[1].split(']')[0])
            cb = int(parts[1].split('[')[1].split(']')[0])
            meas[q] = cb
        else:
            up = line.upper()
            if 'SX' in up:
                n_sx += 1
            if up.startswith('CZ'):
                n_cz += 1
    return meas, n_sx, n_cz


# ============================================================
# 布局工具
# ============================================================
def layout_info():
    """返回 (物理比特列表, 邻接表 {q:[邻居...]}, 排序边清单 [(a,b),...])"""
    ld = json.load(open(LAYOUT))
    phys = ld['physical_qubits']
    adj = {}
    for a, b in ld['edges']:
        p1, p2 = phys[a], phys[b]
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)
    adj = {q: sorted(v) for q, v in adj.items()}
    seen, edges = set(), []
    for q, ns in adj.items():
        for n in ns:
            e = (min(q, n), max(q, n))
            if e not in seen:
                seen.add(e)
                edges.append(e)
    return phys, adj, sorted(edges)


def submit_and_get(compiled_list, tag):
    """一批提交（双开关全关）+ 条数强校验，返回 (task_id, counts 列表)"""
    tid = submit_batch(compiled_list, backend=BACKEND, shots=SHOTS,
                       local_compile=0, cloud_compile=0)
    print('  [%s] task=%s 提交，等待...' % (tag, tid), flush=True)
    res = wait_for_result(tid, timeout=10800)
    items = res if isinstance(res, list) else \
        (res.results if hasattr(res, 'results') else res.get('results'))
    assert len(items) == len(compiled_list), \
        '[%s] 返回 %d != 发出 %d' % (tag, len(items), len(compiled_list))
    print('  [%s] 返回 %d/%d OK' % (tag, len(items), len(compiled_list)), flush=True)
    return tid, [(it.counts if hasattr(it, 'counts') else it.get('counts'))
                 for it in items]


def save(name, obj):
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(obj, open(os.path.join(OUTDIR, name), 'w'),
              indent=1, ensure_ascii=False)


# ============================================================
# 线路构建（单门 level=0 逐门保留；CZ 默认编译每层挡合并）
# ============================================================
def build_1q_tasks(bi, q, neighbor):
    """单比特 80 条：8 档 × 10 串；SX 数=2×门数预检；退化重抽"""
    tasks = []
    for m in M_VALUES:
        for k in range(N_SEQ):
            seed = MASTER_SEED + q * 10000 + m * 100 + k
            while True:
                seq = draw_seq(m, random.Random(seed))
                p_exp = ideal_1q(seq)
                if not is_degenerate(p_exp):
                    break
                seed += 1
            c = Circuit(2)
            for theta, phi, _g in seq:
                c.rphi(0, theta=theta, phi=phi)
            c.measure(0)
            comp = compile(c, backend_info=bi,
                           available_qubits=sorted([q, neighbor]), level=0)
            comp, _ = fix_placement(comp, {0: q})
            meas, n_sx, n_cz = parse_compiled(comp)
            assert n_sx == 2 * m and n_cz == 0 and set(meas) == {q}, \
                'q%d m=%d 预检失败 SX=%d' % (q, m, n_sx)
            tasks.append({'kind': '1q', 'q': q, 'm': m, 'k': k,
                          'p_exp': p_exp, 'compiled': comp})
    return tasks


def build_cz_tasks(bi, edge):
    """边 CZ 80 条：默认编译落边 + SX=2m/CZ=m 预检 + 换位核验（态矢对照）"""
    from uniqc.simulator import Simulator
    a, b = edge
    tasks = []
    for m in M_VALUES:
        for k in range(N_SEQ):
            seed = MASTER_SEED + 500000 + a * 10000 + b * 100 + m * 100 + k
            while True:
                rng = random.Random(seed)
                seq0 = draw_seq(m, rng)
                seq1 = draw_seq(m, rng)
                p_exp = ideal_2q(seq0, seq1)
                if not is_degenerate(p_exp):
                    break
                seed += 1
            c = Circuit(2)
            for i in range(m):
                c.rphi(0, theta=seq0[i][0], phi=seq0[i][1])
                c.rphi(1, theta=seq1[i][0], phi=seq1[i][1])
                c.cz(0, 1)
            c.measure(0)
            c.measure(1)
            comp = compile(c, backend_info=bi, available_qubits=[a, b])
            comp, _ = fix_placement(comp, {0: a, 1: b})
            meas, n_sx, n_cz = parse_compiled(comp)
            assert set(meas) == {a, b} and n_cz == m and n_sx == 2 * m, \
                '边 %s m=%d 预检失败 CZ=%d SX=%d' % (edge, m, n_cz, n_sx)
            psi = np.array(Simulator('statevector').simulate_statevector(comp))
            p_sim = np.abs(psi) ** 2
            tvd_id = 0.5 * float(np.sum(np.abs(p_sim - p_exp)))
            tvd_sw = 0.5 * float(np.sum(np.abs(p_sim[[0, 2, 1, 3]] - p_exp)))
            assert min(tvd_id, tvd_sw) < 0.02, '边 %s m=%d 编译核验失败' % (edge, m)
            q0p = a if tvd_id <= tvd_sw else b
            q1p = b if q0p == a else a
            tasks.append({'kind': 'cz', 'm': m, 'k': k, 'p_exp': p_exp,
                          'compiled': comp, 'cbit0': meas[q0p], 'cbit1': meas[q1p]})
    return tasks


# ============================================================
# 阶段①：读出
# ============================================================
def phase_readout(bi, qubits, adj, submit):
    progs, meta = [], []
    for q in qubits:
        nbr = adj[q][0]                          # 任取一个真实邻居构成合法对
        for kind in ('g0', 'g1'):
            c = Circuit(2)
            if kind == 'g1':                     # X = RPHI(pi, 0)
                c.rphi(0, theta=3.141592653589793, phi=0.0)
            c.measure(0)
            comp = compile(c, backend_info=bi, available_qubits=sorted([q, nbr]),
                           level=0)
            comp, _ = fix_placement(comp, {0: q})
            meas, n_sx, n_cz = parse_compiled(comp)
            assert set(meas) == {q} and n_cz == 0, 'q%d 读出线路异常' % q
            if kind == 'g1':
                assert n_sx == 2, 'q%d X 门应=2 SX' % q
            progs.append(comp)
            meta.append((q, kind))
    print('[① 读出] 构建 %d 条（13 比特 × 2）' % len(progs), flush=True)
    if not submit:
        return
    tid, counts_list = submit_and_get(progs, 'readout')
    per_q = {}
    for (q, kind), counts in zip(meta, counts_list):
        dist = counts_to_dist(counts, 1, 0)
        per_q.setdefault(q, {})
        if kind == 'g0':
            per_q[q]['e0'] = float(dist[1])      # |0> 读成 1
        else:
            per_q[q]['e1'] = float(dist[0])      # |1> 读成 0
            per_q[q]['x_flip_P1'] = float(dist[1])
    for q in qubits:
        d = per_q[q]
        d['e_q'] = (d['e0'] + d['e1']) / 2
        if d['x_flip_P1'] <= 0.8:
            # 2026-10-08 用户拍板：翻转核验降为软警（仅记录不干涉）
            # ——P1>0.5 即 X 门生效；偏低只反映该比特 e1 大，如实进表
            print('  [软警] q%d X 翻转 P1=%.3f（<0.8，e1 偏大，记录不拦截）'
                  % (q, d['x_flip_P1']))
        d['x_flip_ok'] = d['x_flip_P1'] > 0.5   # 硬底线：>0.5 才算 X 真生效
        print('  q%-3d e0=%.4f e1=%.4f e_q=%.4f' % (q, d['e0'], d['e1'], d['e_q']))
    save('readout_all.json', {
        'shots': SHOTS, 'task_ids': [tid],
        'per_qubit': {str(q): per_q[q]['e_q'] for q in qubits},
        'detail': {str(q): per_q[q] for q in qubits}})


# ============================================================
# 阶段②：单门（逐比特 80 条一小批）
# ============================================================
def phase_single(bi, qubits, adj, submit):
    per_q, task_ids = {}, []
    for q in qubits:
        tasks = build_1q_tasks(bi, q, adj[q][0])
        print('[② 单门] q%d 构建 80 条' % q, flush=True)
        if not submit:
            continue
        tid, counts_list = submit_and_get([t['compiled'] for t in tasks],
                                          '1q_q%d' % q)
        task_ids.append(tid)
        per_m = {}
        for t, counts in zip(tasks, counts_list):
            dist = counts_to_dist(counts, 1, 0)
            f = fxeb_neill(dist, t['p_exp'], D1)
            if f is not None:
                per_m.setdefault(t['m'], []).append(f)
        ms = sorted(per_m)
        means = [float(np.mean(per_m[m])) for m in ms]
        assert means[0] - means[-1] >= DECAY_GUARD, \
            'q%d 兜底判据：无明显衰减 %s' % (q, np.round(means, 3))
        per_q[str(q)] = fit_eps(ms, means, D1)
        print('  q%-3d eps1=%.5f（F(2)=%.3f -> F(500)=%.3f）'
              % (q, per_q[str(q)], means[0], means[-1]), flush=True)
        save('single_all.json', {'shots': SHOTS, 'task_ids': task_ids,
                                 'per_qubit': per_q})     # 逐批增量落盘
    if submit:
        print('[② 单门] %d 比特完成' % len(per_q), flush=True)


# ============================================================
# 阶段③：双门（逐边 240 条一小批：CZ 80 + 两端单门 160）
# ============================================================
def phase_cz(bi, edges, submit):
    per_edge, eps2c_all, e1_batch, task_ids = {}, {}, {}, []
    for edge in edges:
        a, b = edge
        tasks = build_cz_tasks(bi, edge)
        # 两端单门同批（与 CZ 同批次口径；邻居=对方端点）
        tasks += build_1q_tasks(bi, a, b)
        tasks += build_1q_tasks(bi, b, a)
        print('[③ 双门] 边 %d-%d 构建 %d 条' % (a, b, len(tasks)), flush=True)
        if not submit:
            continue
        tid, counts_list = submit_and_get([t['compiled'] for t in tasks],
                                          'cz_%d-%d' % (a, b))
        task_ids.append(tid)
        cz_rows, q_rows = {}, {}
        for t, counts in zip(tasks, counts_list):
            if t['kind'] == 'cz':
                dist = counts_to_dist(counts, 2, t['cbit0'], t['cbit1'])
                f = fxeb_neill(dist, t['p_exp'], D4)
                if f is not None:
                    cz_rows.setdefault(t['m'], []).append(f)
            else:
                dist = counts_to_dist(counts, 1, 0)
                f = fxeb_neill(dist, t['p_exp'], D1)
                if f is not None:
                    q_rows.setdefault(t['q'], {}).setdefault(t['m'], []).append(f)
        # CZ 曲线 -> eps2c
        ms_cz = sorted(cz_rows)
        means_cz = [float(np.mean(cz_rows[m])) for m in ms_cz]
        assert means_cz[0] - means_cz[-1] >= DECAY_GUARD, \
            '边 %s CZ 无衰减' % (edge,)
        eps2c = fit_eps(ms_cz, means_cz, D4)
        # 两端单门（同批）-> eps1
        e1_local = {}
        for q in (a, b):
            per_m = q_rows[q]
            ms = sorted(per_m)
            means = [float(np.mean(per_m[mm])) for mm in ms]
            assert means[0] - means[-1] >= DECAY_GUARD, \
                '边 %s 端 q%d 单门无衰减' % (edge, q)
            e1_local[q] = fit_eps(ms, means, D1)
        eps2 = eps2c - e1_local[a] - e1_local[b]
        per_edge['%d-%d' % (a, b)] = eps2
        eps2c_all['%d-%d' % (a, b)] = eps2c
        e1_batch['%d-%d' % (a, b)] = {str(a): e1_local[a], str(b): e1_local[b]}
        print('  边 %d-%d eps2c=%.5f eps1(a)=%.5f eps1(b)=%.5f eps2=%.5f'
              % (a, b, eps2c, e1_local[a], e1_local[b], eps2), flush=True)
        save('cz_all.json', {'shots': SHOTS, 'task_ids': task_ids,
                             'per_edge': per_edge, 'eps2c': eps2c_all,
                             'e1_batch': e1_batch})      # 逐批增量落盘


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--phase', choices=['readout', 'single', 'cz', 'all'],
                    default='all')
    args = ap.parse_args()

    qubits, adj, edges = layout_info()
    print('布局：%d 比特 %s' % (len(qubits), qubits))
    print('边（%d 条）：%s' % (len(edges), edges), flush=True)
    bi = find_backend(BACKEND)

    if args.dry_run or not args.submit:
        phase_readout(bi, qubits, adj, submit=False)
        phase_single(bi, qubits[:1], adj, submit=False)
        phase_cz(bi, edges[:1], submit=False)
        print('ALL DONE (dry-run)：读出 26 + 单门首比特 80 + 首边 240 构建自检全过，未提交')
        return

    if args.phase in ('readout', 'all'):
        phase_readout(bi, qubits, adj, submit=True)
    if args.phase in ('single', 'all'):
        phase_single(bi, qubits, adj, submit=True)
    if args.phase in ('cz', 'all'):
        phase_cz(bi, edges, submit=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
