"""Project entrypoint for the training pipeline.

The implementation lives in training/train_criteo_profit_uplift.py. This file
keeps the root command short:

    python criteo_profit_uplift.py
"""

from training.train_criteo_profit_uplift import main


if __name__ == "__main__":
    main()
