# 随机线路采样（RCS / XEB）实验复现

基于本源量子 uniqc 代码库与 WK_C180 超导量子芯片的随机线路采样实验，
通过线性交叉熵基准（XEB）测量多比特随机线路的保真度，并与理论预测对比。

## 目录结构

```
├── simulator/     # 模拟器实验代码（报告主体）
└── real_machine/  # 真机实验源文件（WK_C180）
```

### simulator/ —— 模拟器实验代码

| 文件 | 用途 |
|---|---|
| `xeb_v2.py` | XEB 实验主程序（模拟器版）：随机线路生成 → 无噪/含噪演化 → 采样 → F_XEB |
| `calibrate_v2_pseudocode_2.0_isolated_neill.py` | 标定代码：单门/双门错误率标定 + 读出标定 + 预测保真度（4 模块） |
| `layout_gen.py` | 量子比特布局生成：任意 n 的错位网格 + ABCD 四色边染色 |
| `标定实验步骤.md` | 标定实验操作流程文档 |
| `实验结果汇总.md` | 模拟器阶段实验结果记录 |

### real_machine/ —— 真机实验源文件（WK_C180）

| 文件 | 用途 |
|---|---|
| `xeb_v2_real.py` | XEB 实验主程序（真机版）：限池编译、批量提交、结果解析、新口径 α 预测 |
| `layout_gen.py` | 布局生成 + 芯片拓扑选比特（含数据洞过滤） |
| `chip_data.py` | 工作台 XLSX 标定文件 → 芯片参数 JSON 转换器 |
| `selected_layout_n13_v2.json` | 当前实验布局：n=13 三环簇（13 比特 / 15 边 / 3 六边形 / 4-4-4-3 染色） |
| `chip_params_WK_C180.json` | WK_C180 芯片标定参数（由 XLSX 转换，2026-09-21） |
| `tests.py` | 主程序依赖的自检模块 |

> 注：两目录各有一份 layout_gen.py：simulator/ 为原代码仓历史版本（原样保留），real_machine/ 为现役版本（含数据洞过滤补丁），运行真机实验请使用后者。

## 运行方式

### 真机深度扫描实验（n=13 三环簇）

```bash
export ORIGINQ_API_KEY=<你的密钥>
cd real_machine
python3 xeb_v2_real.py --real \
    --backend originq:WK_C180 \
    --layout selected_layout_n13_v2.json \
    --chip-params ../real_machine/chip_params_WK_C180.json \
    --ms 5,10,15,20,25 --k 20 --seed-base 70000 \
    --shots 2000 --batch
# 加 --dry-run 只做编译自检与 α 计算，不提交任务
```

### 模拟器实验

```bash
cd simulator
python3 xeb_v2.py --n 20 --noise      # 含噪模拟
python3 xeb_v2.py --n 20              # 无噪验证（F 应≈1）
python3 calibrate_v2_pseudocode_2.0_isolated_neill.py   # 标定
```

## α 预测公式（真机版，2026-09-23 定稿）

```
α = ∏(1−e_q) × ∏F1q(q)^m × ∏F_CZ(e)^n_e × ∏( 2·e^(−T总/T2*) + e^(−T总/T1) ) / 3
```

- ① 读出项：每比特乘 1 次（e_q = 1 − 读出保真度）
- ② 单门项：每比特乘 m 次（保真度直接连乘，无换算）
- ③ 双门项：每条边乘其被 CZ 的次数（按 ABCDCDAB 逐色计数）
- ④ 退相干项：电路总运行时长 T_总（编译调度表实测），T1/T2* 逐比特取自标定

## 依赖

- Python 3.10+
- uniqc（本源量子云 SDK）
- pyqpanda3、numpy、networkx、matplotlib、scipy

## 安全提示

本仓库不含任何 API 密钥。运行真机实验前自行设置 `ORIGINQ_API_KEY` 环境变量。
