# Training 目录说明

这里放正式的训练与评估代码，主入口是：

```bash
python training/train_criteo_profit_uplift.py
```

根目录的 `criteo_profit_uplift.py` 也可以直接运行，真正的实现都在这个目录里。

这条训练流水线做几件事：

```text
1. 读取 data/ 里的 50 万行数据快照；没有快照时自动构建同结构数据
2. 训练 CVR、GMV、T-Learner、DR-Learner 几类策略模型
3. 用 cross-fitting 构造 DR-Learner 的 pseudo outcome
4. 按 predicted_tau_profit / coupon_cost 做预算约束投放
5. 输出策略对比、预算曲线、Qini/AUUC、分层校准和人群诊断
6. 把训练好的模型保存到 models/
```

常用命令：

```bash
python training/train_criteo_profit_uplift.py --backend sklearn --output-dir outputs_criteo --model-dir models
```

快速跑小样本：

```bash
python training/train_criteo_profit_uplift.py --sample-size 50000 --budget 3000 --n-folds 3
```

如果要接真实 Criteo 文件：

```bash
python training/train_criteo_profit_uplift.py --data-path data/criteo-uplift-v2.1.csv.gz
```

默认使用 sklearn 后端，保证压缩包在常见环境里能复现同一批结果；如果环境安装了 LightGBM，可以显式传 `--backend lightgbm`。
