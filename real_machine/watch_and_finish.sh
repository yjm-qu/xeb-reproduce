#!/bin/bash
# watch_and_finish.sh —— 盯标定合并批，返回后全自动收尾（2026-10-10）
set -e
cd /home/ubuntu/RCS_fuxian/xeb_v2_with_layout

echo "[监测] 盯标定合并批 uqt_ac5a8d34..."
for i in $(seq 1 900); do
  ST=$(python3 - << 'EOF' 2>/dev/null
import os
os.environ['ORIGINQ_API_KEY'] = open('/home/ubuntu/.originq_key').read().strip()
import uniqc.backend_adapter.task_manager as tm
print(tm.query_task('uqt_ac5a8d349c534870a6fc1291394dc59a').status)
EOF
)
  echo "[监测] $(date +%H:%M:%S) 状态: $ST"
  if [ "$ST" = "success" ] || [ "$ST" = "finished" ]; then
    echo "[监测] 完成，开始取数+灌表+算α..."
    break
  fi
  if echo "$ST" | grep -qi 'fail\|error'; then
    echo "[监测] 任务失败！保留现场停止" | tee results/calib_full_13/merge_fail.log
    exit 1
  fi
  sleep 120
done

python3 fetch_all.py calib 2>&1 | tee results/calib_full_13/fetch_merge.log | grep -v Warning
CALIB_EDGES=15 python3 build_measured_params_v2.py 2>&1 | tee results/calib_full_13/build_table.log | tail -8

echo "[收尾] 用实测表算 30 条主实验的 α..."
python3 - << 'EOF' 2>&1 | tee results/alpha_report.log
import json
import numpy as np
import os, sys
sys.path.insert(0, '.')
os.environ['ORIGINQ_API_KEY'] = open('/home/ubuntu/.originq_key').read().strip()
import xeb_v2_real as XR

ld = XR.load_layout_from_json('selected_layout_n13A_3col.json', 0)
chip = XR.load_chip_params('measured_params.json')
main = json.load(open('results/main_exp_result.json'))
alpha = XR.compute_alpha_f_exact(ld, chip, 13, main['m'], ld['color_assignment'])[0]
fs = [r['F'] for r in main['per_seed']]
fm = float(np.mean(fs)); se = float(np.std(fs, ddof=1)/np.sqrt(len(fs)))
print('===== 最终对比（实测表口径） =====')
print('F  = %.4f ± %.4f（30 条，m=21）' % (fm, se))
print('α  = %.6f（实测表：simultaneous 单门+负值截0）' % alpha)
print('F/α = %.2f' % (fm/alpha) if alpha > 1e-6 else 'α≈0：两者同为地板值，一致')
sig = (fm - alpha)/se if se > 0 else 0
print('σ 偏离 = %+.1f' % sig)
out = {'F_mean': fm, 'F_SEM': se, 'alpha_measured': alpha, 'sigma': sig,
       'task_main': 'uqt_07592712e7dc4deaa61c03055f2e23e8',
       'task_calib': 'uqt_ac5a8d349c534870a6fc1291394dc59a'}
json.dump(out, open('results/final_comparison.json', 'w'), indent=1)
print('saved results/final_comparison.json')
EOF
echo "[监测] ALL DONE $(date)"
