# image-binary-clf

用 PyTorch 从零搭一个小 CNN，做**洛天依 vs 初音未来**的图像二分类。
起因是《人工智能概论》第一次作业（分类器），顺带把 PyTorch 的训练/评估/落盘闭环走了一遍。

> ⚠️ **本仓库不含任何图片数据**。数据需自行准备（见下方「数据准备」），且仅限个人学习用途。

## 结果

| 项 | 值 |
|---|---|
| 数据规模 | 每类 1200 张（train 960 / val 120 / test 120） |
| 模型 | 自建 `TinyCNN`（2 层卷积 + 全局平均池化，**无预训练**） |
| 最好 val F1 | 0.754 |
| **test acc / F1** | **0.717 / 0.736** |

⭐ 一个比分数更值得看的现象：**训练轮数从 10 加到 35，test 准确率反而从 0.725 掉到 0.717**。
train loss 一路降到 0.54、val loss 却卡在 0.56 并开始翘——这个模型容量已经到顶，加轮数只是把训练集背得更熟。
错误样本分析显示，模型主要在学"主色调统计"（两者都是青绿/蓝白系），
遇到换发色的角色、非立绘的舞台照、多角色同框图就失效。

## 结构

```
.
├── common.py              模型 / 预处理 / 指标 / 训练评估函数（train 与 eval 共用）
├── train.py               训练：新开任务 / 续训 / 指定轮数
├── eval.py                测试：best 模型在 test 集上的指标
├── viz.py                 可视化：训练曲线 / 混淆矩阵 / 错误样本 / 阈值扫描
├── notebooks/
│   └── 01_image_binary_clf_pipeline.ipynb   最初的学习过程（逐单元讲解版）
├── scripts/data/          数据准备流水线（采集 → 去重 → 划分 → 复制 → 预处理）
│   ├── 01_crawl_miku.py
│   ├── 02_dedup.py
│   ├── 03_split.py
│   ├── 04_copy_to_process.py
│   └── 05_preprocess.py
└── pyproject.toml / uv.lock
```

数据目录约定（都是相对路径，可通过命令行参数覆盖）：

```
data/
├── raw/{luo,miku}/        原始图集
├── splits.csv             划分清单（按作品号/去重组隔离，8:1:1 分层）
└── process/{train,val,test}/{luo,miku}/   预处理后的 jpg（长边 ≤1024）
```

## 快速开始

```bash
uv sync

# 1) 数据准备（需自备图集，放到 data/raw/{luo,miku}/）
uv run python scripts/data/01_crawl_miku.py --out data/raw/miku      # 采集（可选，若已有图集）
uv run python scripts/data/02_dedup.py --dir data/raw/luo \
    --out-md data/_dedup_report.md --out-csv data/_dedup_groups.csv  # MD5 + pHash 去重
uv run python scripts/data/03_split.py --raw-root data/raw --out data/splits.csv
uv run python scripts/data/04_copy_to_process.py
uv run python scripts/data/05_preprocess.py

# 2) 训练
uv run python train.py --epochs 10 --data-dir data/process
uv run python train.py --resume latest --epochs 5            # 接着上次再跑 5 轮
uv run python train.py --resume latest --epochs 5 --lr 3e-4  # 续训并降学习率

# 3) 测试 + 出图
uv run python eval.py --data-dir data/process
uv run python viz.py --sweep
```

## 产物

每个 run 一个目录 `runs/<YYYYMMDD_HHMMSS>/`（已 gitignore）：

```
config.json          超参 + 预处理参数 + 类别映射（推理靠它还原 transform）
metrics.csv          每轮一行；续训时追加，不覆盖历史
last_ckpt.pt         model + optimizer + epoch + best_val_f1（续训用）
best_model.pt        验证集 F1 最高时的纯 state_dict（推理用）
test_metrics.csv     eval.py 写出的测试结果
figs/                viz.py 出的四张图
```

## 几个刻意的选择

- **不做 `Resize(224,224)`**：会把 3:4 / 16:9 的画幅信息抹掉、人物拉变形。用短边缩放 256 + 中心裁剪。
- **单 logit + `BCEWithLogitsLoss`**：不在模型里加 Sigmoid，数值更稳。
- **续训连 optimizer 状态一起存**：只存 model 等于把 Adam 的一阶/二阶矩清零，等于热启动新优化器。
- **best 按 val F1 选**，不按 acc——两类配平，但混淆方向不对称，F1 更能反映真实水平。
- **train / eval / viz 共用 `common.py`**：模型结构和预处理各写一份，改一处忘一处就会出现"权重能加载、指标全错"的静默 bug。

## 关于数据与版权

洛天依（上海禾念）与初音未来（Crypton Future Media）的角色版权归各自权利方所有。
本仓库**不分发任何图片**，`scripts/data/` 里的采集脚本仅用于个人学习时抓取公开图集；
`data/` 已被 gitignore。若复用这些代码，请自行确认抓取与使用行为的合规性，不要用于商业用途。

## License

MIT © ttzc
