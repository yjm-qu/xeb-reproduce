# 随机线路采样（RCS / XEB）实验复现

基于本源量子 uniqc 代码库与 WK_C180 超导量子芯片的随机线路采样实验，
通过线性交叉熵基准（XEB）测量多比特随机线路的保真度，并与理论预测对比。

## 目录结构

```
├── simulator/     # 模拟器实验代码（报告主体）
└── real_machine/  # 真机实验源文件（WK_C180）
    └── results/   # 真机实验结果数据（标定三件套 + 主实验 + 对比表）
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

**主程序与基础设施**

| 文件 | 用途 |
|---|---|
| `xeb_v2_real.py` | XEB 实验主程序（真机瘦身版）：ABCACB 三色轮转线路构建、批量提交、cbit→逻辑映射解析、F_XEB 与 α 对比 |
| `tests.py` | 主程序依赖的自检模块 |
| `layout_gen.py` | 布局生成 + 芯片拓扑选比特（含数据洞过滤） |
| `chip_data.py` | 工作台 XLSX 标定文件 → 芯片参数 JSON 转换器（按后端区分断言规模） |
| `chip_params_WK_C180.json` | WK_C180 芯片官方标定参数（XLSX 转换，2026-10-08 校准） |
| `calibrate_real.py` | 标定辅助：定点锁比特放置（fix_placement）等 |

**实测标定流水线（2026-10-08 章程，shots=20000）**

| 文件 | 用途 |
|---|---|
| `calib_full_13.py` | 三项错误率标定铺开：①读出 26 条 ②单门 13×8 档×10 串 ③双门 15 边×（CZ+两端单门）；`--dry-run` 自检 / `--submit --phase xxx` 断点续跑 |
| `build_measured_params_v2.py` | 实测表生成：单门取 simultaneous 口径（边批内嵌单门均值），双门负值截 0 标注；产出 `measured_params.json` |
| `measured_params.json` | 实测标定表（与官方表同构格式，含溯源 provenance） |


**布局**

| 文件 | 用途 |
|---|---|
| `selected_layout_n13A_3col.json` | **现役布局**：n=13 三环簇 A 版 3 色（物理比特 94~141 段，15 边，染色 A/B/C 各 5） |
| `selected_layout_n13_v2.json` | 旧 4 色布局（已弃用，留档） |

### real_machine/results/ —— 真机实验结果

| 文件 | 内容 |
|---|---|
| `calib_full_13/readout_all.json` | 读出标定：13 比特 e_0/e_1/e_q |
| `calib_full_13/single_all.json` | 单门标定：13 比特 × 8 档深度衰减拟合 ε₁ |
| `calib_full_13/cz_all.json` | 双门标定：15 边 ε₂ᶜ 与拆账 ε₂ |
| `main_exp_result.json` | 主实验 m=21×K=30：逐线路 F_XEB 与汇总 |
| `shallow_result.json` | 浅档验证 m=9/12/15/18×K=30 |
| `alpha_two_tables.json` | 双表对比：官方表 α vs 实测表 α（各深度） |
| `final_comparison.json` | 定稿对比（不剔除口径，表 A） |
| `all_F_raw.json` | 全部原始 F 值 |
| `shallow_iqr.json` | IQR 剔除对照表（含各档剔除数与占比） |

> 注：simulator/ 与 real_machine/ 各有一份 layout_gen.py：前者为原代码仓历史版本（原样保留），后者为现役版本。

## 运行方式

### 真机实验全流程（标定 → 灌表 → 主实验）

```bash
export ORIGINQ_API_KEY=<你的密钥>
cd real_machine

# ① 三项错误率标定（先 dry-run 自检）
python3 calib_full_13.py --dry-run
python3 calib_full_13.py --submit

# ② 实测表生成（防呆三查：齐全性/合法性/溯源）
python3 build_measured_params_v2.py

# ③ 主实验（m=21 × K=30，ABCACB 三色轮转）
python3 xeb_v2_real.py --real --backend originq:WK_C180 \
    --layout selected_layout_n13A_3col.json \
    --chip-params measured_params.json \
    --ms 21 --k 30 --shots 20000 --seed-base 90000
# 加 --dry-run 只做编译自检与 α 计算，不提交任务；--self-test 跑单元测试
```

### 模拟器实验

```bash
cd simulator
python3 xeb_v2.py --n 20 --noise      # 含噪模拟
python3 xeb_v2.py --n 20              # 无噪验证（F 应≈1）
python3 calibrate_v2_pseudocode_2.0_isolated_neill.py   # 标定
```

## α 预测公式（2026-10-10 定稿，无④退相干项）

```
α = ∏(1−e_q) × ∏F1q(q)^m × ∏F_CZ(e)^n_e
```

- ① 读出项：每比特乘 1 次（e_q 实测 = (e_0+e_1)/2）
- ② 单门项：每比特乘 m 次（保真度直接连乘）
- ③ 双门项：每条边乘其被 CZ 的次数（ABCACB 轮转逐色计数）

## 第三轮实验结果摘要（2026-10-08 ~ 10-10）

- 布局：13 比特三环簇 A 版 3 色（ABCACB 消除同色连拍），shots=20000 全流程统一
- 标定：读出 e_q 均值 0.59%；单门 ε₁（simultaneous 口径）；双门 ε₂ᶜ 拆账（3/15 边负值截 0，isolated≠simultaneous 实证）
- 主实验：m=21 实测 F = −0.0030 ± 0.0022（信号已耗尽）；浅档 m=9/12/15/18 检验 F/α
- 双表对比：官方 XLSX 表 α 高估实测 F 27~55 倍；实测标定表 α 高估 3~4 倍（改善约一个数量级）
- 详细数据见 `real_machine/results/`

## 依赖

- Python 3.10+
- uniqc（本源量子云 SDK）
- pyqpanda3、numpy、networkx、matplotlib、scipy

## 安全提示

本仓库不含任何 API 密钥。运行真机实验前自行设置 `ORIGINQ_API_KEY` 环境变量
（本机若有 `~/.originq_key` 密钥文件，脚本会自动加载）。
