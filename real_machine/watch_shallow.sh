#!/bin/bash
# watch_shallow.sh —— 盯浅档批，返回后自动取数+按档算F+与α对比（2026-10-10）
set -e
cd /home/ubuntu/RCS_fuxian/xeb_v2_with_layout

TID=uqt_5da56f8d707545cc9abebb251ad980fc
echo "[监测] 盯浅档批 $TID ..."
for i in $(seq 1 900); do
  ST=$(python3 - << EOF 2>/dev/null
import os
os.environ['ORIGINQ_API_KEY'] = open('/home/ubuntu/.originq_key').read().strip()
import uniqc.backend_adapter.task_manager as tm
print(tm.query_task('$TID').status)
EOF
)
  echo "[监测] $(date +%H:%M) 状态: $ST"
  if [ "$ST" = "success" ] || [ "$ST" = "finished" ]; then
    echo "[监测] 完成，取数+分析..."
    break
  fi
  if echo "$ST" | grep -qi 'fail\|error'; then
    echo "[监测] 失败！保留现场" | tee results/shallow_fail.log
    exit 1
  fi
  sleep 120
done

python3 - << 'EOF' 2>&1 | tee results/shallow_report.log | grep -v Warning
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
os.environ['ORIGINQ_API_KEY'] = open('/home/ubuntu/.originq_key').read().strip()
from uniqc import wait_for_result
import xeb_v2_real as XR

P = json.load(open('pending_shallow.json'))
res = wait_for_result(P['task_id'], timeout=300)
items = res if isinstance(res, list) else (res.results if hasattr(res, 'results') else res.get('results'))
assert len(items) == P['n_circuits']
counts_list = [(it.counts if hasattr(it, 'counts') else it.get('counts')) for it in items]
print('取回 %d/%d 条' % (len(counts_list), P['n_circuits']))

ld = XR.load_layout_from_json('selected_layout_n13A_3col.json', 0)
chip = XR.load_chip_params('measured_params.json')
per_m = {}
for j, counts in zip(P['jobs'], counts_list):
    psi = np.array([complex(r, i) for r, i in zip(j['psi']['re'], j['psi']['im'])])
    norm = {int(k): v for k, v in j['norm'].items()}
    bits = []
    for kk_, cnt in counts.items():
        s = str(kk_).strip()
        if s[:2].lower() == '0b':
            s = s[2:]
        cidx = int(s, 2) if set(s) <= {'0', '1'} else int(s)
        logical = 0
        for b in range(13):
            if (cidx >> b) & 1:
                logical |= (1 << norm.get(b, b))
        bits.extend([logical] * cnt)
    F = XR.compute_F_XEB(psi, bits, 13)
    per_m.setdefault(j['m'], []).append(float(F))

print()
print('===== 浅档验证（实测表口径） =====')
print('| m | K | F 均值±SEM | α | F/α | 偏离σ |')
print('|---|---|---|---|---|---|')
rows_out = []
for m in sorted(per_m):
    fs = np.array(per_m[m])
    fm, se = fs.mean(), fs.std(ddof=1)/np.sqrt(len(fs))
    alpha = XR.compute_alpha_f_exact(ld, chip, 13, m, ld['color_assignment'])[0]
    ratio = fm/alpha if alpha > 1e-9 else float('nan')
    sig = (fm - alpha)/se if se > 0 else float('nan')
    print('| %d | %d | %.4f±%.4f | %.5f | %.2f | %+.1f |' % (m, len(fs), fm, se, alpha, ratio, sig))
    rows_out.append({'m': m, 'K': len(fs), 'F_mean': round(float(fm),4),
                     'F_SEM': round(float(se),4), 'alpha': round(float(alpha),5),
                     'ratio': round(float(ratio),3), 'sigma': round(float(sig),2)})
json.dump({'task_id': P['task_id'], 'rows': rows_out},
          open('results/shallow_result.json', 'w'), indent=1)
print('saved results/shallow_result.json')
EOF
echo "[监测] ALL DONE $(date)"
