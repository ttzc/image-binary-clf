# image-binary-clf

洛天依 vs 初音未来 · 图像二分类（AI 概论第一次作业）。
学习项目：借这次作业把 PyTorch 的训练/评估/落盘闭环走一遍。

数据：`D:/image/process_1024/{train,val,test}/{luo,miku}/`（每类 1200 张，划分见 `D:/image/splits.csv`）

## 文件

| 文件 | 作用 |
| --- | --- |
| `common.py` | 共用定义：`TinyCNN`、transforms、指标、训练/推理函数。**train 与 eval 必须共用**，否则模型结构和预处理会漂移 |
| `train.py` | 训练：新开任务 / 续训 / 指定轮数 |
| `eval.py` | 测试：best 模型在 test 集上的完整指标 |
| `01_image_binary_clf_pipeline.ipynb` | 最初的学习过程（逐单元讲解版），脚本是从这里抽出来的 |

## 用法

```bash
# 新开一个任务，跑 10 轮
uv run python train.py --epochs 10

# 接着最近一次训练的 last_ckpt 再跑 5 轮
uv run python train.py --resume latest --epochs 5

# 指定某个 run 续训，并降低学习率
uv run python train.py --resume runs/20261102_214648 --epochs 5 --lr 3e-4

# 评估最近一次 run 的 best 模型
uv run python eval.py

# 换阈值 + 扫一遍最优阈值
uv run python eval.py --threshold 0.6 --sweep
```

`--epochs` 的语义：**本次要跑多少轮**。续训时表示「在已训练轮数之后再跑 N 轮」，
所以 `--resume latest --epochs 5` 是在 10 轮基础上去到 15 轮。

## 产物

每个 run 一个目录 `runs/<YYYYMMDD_HHMMSS>/`：

```
config.json          超参 + 预处理参数 + 类别映射（推理靠它还原 transform）
metrics.csv          每轮一行；续训时追加，不覆盖历史
last_ckpt.pt         model + optimizer + epoch + best_val_f1（续训用）
best_model.pt        验证集 F1 最高时的纯 state_dict（推理用）
test_metrics.csv     eval.py 写出的测试结果
training_curves.png  notebook 里那版画出的曲线
```

`runs/` 与 `*.pt` 已在 `.gitignore` 中忽略。

## 几个刻意的选择

- **不做 `Resize(224,224)`**：会把 3:4 / 16:9 的画幅信息抹掉、人物拉变形。用短边缩放 256 + 中心裁剪。
- **单 logit + `BCEWithLogitsLoss`**：不在模型里加 Sigmoid，数值更稳。
- **续训连 optimizer 状态一起存**：只存 model 等于把 Adam 的一阶/二阶矩清零，等于热启动。
- **best 按 val F1 选**，不按 acc——两类配平，但混淆方向不对称，F1 更能反映真实水平。
