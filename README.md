# 固定预算下的优惠券增量利润优化

这个项目解决的是一个营销投放问题：每周有一批优惠券和固定预算，要决定把券发给哪些用户。这里我没有把目标设成“预测谁会转化”，而是直接优化 **增量利润**：

```text
tau_profit(x) = E[profit | treatment=1, x] - E[profit | treatment=0, x]
```

投放时按下面的分数排序，直到预算用完：

```text
score = predicted_tau_profit / coupon_cost
```

这样做的出发点是：高转化用户不一定值得发券，因为很多人本来就会买；真正应该找的是“被券改变行为、且扣掉券成本后仍有增量利润”的用户。

## 目录结构

```text
training/       训练和评估代码
data/           50 万行可复现数据快照
models/         训练好的模型和 model_card
outputs_criteo/ 正式实验结果
```

主入口在：

```text
training/train_criteo_profit_uplift.py
```

根目录也保留了 `criteo_profit_uplift.py`，方便直接运行。

## 数据

数据结构参考 Criteo Uplift Prediction Dataset：

```text
f0 ... f11, treatment, conversion, visit, exposure
```

公开 Criteo 数据没有用户级利润和反事实利润，所以我在同样的 schema 上补了一层半仿真的业务变量：

```text
coupon_cost, gmv, profit, tau_profit
```

项目里已经保存了默认数据快照：

```text
data/criteo_profit_uplift_sample500000_seed42.csv.gz
```

因此评审拿到文件夹后可以直接复现，不需要重新下载数据。如果本地有原始 Criteo 文件，可以通过 `--data-path` 指定；如果要重建数据快照，可以加 `--no-cache-data`。

## 模型

我对比了五种策略：

```text
Random
CVR Targeting
GMV Targeting
T-Learner
DR-Learner
```

主方案是 `DR-Learner`。为了保证压缩包在常见 Python 3.8+ 环境里能稳定复现，默认后端固定为 sklearn；如果评审环境安装了 LightGBM，可以加 `--backend lightgbm` 跑增强版。

DR-Learner 先估计：

```text
mu1(x) = E[profit | treatment=1, x]
mu0(x) = E[profit | treatment=0, x]
e(x)   = P(treatment=1 | x)
```

然后用 cross-fitting 构造 doubly robust pseudo outcome：

```text
tau_dr = mu1 - mu0
       + W * (Y - mu1) / e
       - (1-W) * (Y - mu0) / (1-e)
```

我选择 DR-Learner 的原因是，营销历史数据里的发券通常不是随机的；用 outcome model 和 propensity model 同时校正，比单纯做响应预测更稳。

## 评估

主要指标包括：

```text
incremental_profit
ROI
AUUC / Qini
budget sensitivity
uplift decile calibration
segment diagnostics
```

除了半仿真的真实 `tau_profit`，我还输出了 AIPW / doubly robust policy value estimate 和 bootstrap 置信区间，模拟真实离线评估时会看的策略价值。

默认 50 万行结果摘要：

```text
DR-Learner incremental_profit = 5088.24
DR-Learner ROI = 0.2035
DR-Learner AIPW estimate = 5214.77
AIPW 95% CI = [4558.61, 5850.88]

T-Learner incremental_profit = 5065.91
CVR Targeting incremental_profit = 1470.95
GMV Targeting incremental_profit = 859.87
Random incremental_profit = -470.26
```

可以看到，CVR/GMV 模型虽然能找到“高价值用户”，但不一定能找到“高增量用户”；DR-Learner 更贴近这个业务问题本身。

## 复现

安装依赖：

```bash
pip install -r requirements.txt
```

如果要运行 LightGBM 版本，再额外安装：

```bash
pip install -r requirements-lightgbm.txt
```

正式运行：

```bash
python training/train_criteo_profit_uplift.py --backend sklearn --output-dir outputs_criteo --model-dir models
```

快速调试：

```bash
python training/train_criteo_profit_uplift.py --sample-size 50000 --budget 3000 --n-folds 3
```

接入本地 Criteo 文件：

```bash
python training/train_criteo_profit_uplift.py --data-path data/criteo-uplift-v2.1.csv.gz
```

如果要使用 LightGBM：

```bash
python training/train_criteo_profit_uplift.py --backend lightgbm --output-dir outputs_criteo --model-dir models
```

主要结果文件：

```text
outputs_criteo/strategy_comparison.csv
outputs_criteo/budget_curve.csv
outputs_criteo/uplift_decile_calibration.csv
outputs_criteo/segment_analysis.csv
outputs_criteo/qini_curve.png
outputs_criteo/budget_profit_curve.png
models/model_card.json
models/dr_tau_model.pkl
```

完整提交清单见 `SUBMISSION_MANIFEST.md`。
