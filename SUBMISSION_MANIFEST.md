# 提交清单

正式提交时建议只打包下面这些内容：

```text
README.md
requirements.txt
requirements-lightgbm.txt
criteo_profit_uplift.py
training/
data/
models/
outputs_criteo/
```

各目录含义：

```text
training/       训练和评估代码
data/           默认 50 万行可复现数据快照
models/         已训练模型和 model_card.json
outputs_criteo/ 正式实验结果，包括 CSV 和 PNG
```

推荐复现命令：

```bash
python training/train_criteo_profit_uplift.py --backend sklearn --output-dir outputs_criteo --model-dir models
```

如果评审只想看结果，优先打开：

```text
outputs_criteo/strategy_comparison.csv
outputs_criteo/budget_curve.csv
outputs_criteo/uplift_decile_calibration.csv
models/model_card.json
```
