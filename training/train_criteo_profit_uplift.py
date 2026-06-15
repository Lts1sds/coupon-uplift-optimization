"""Training pipeline for profit uplift modeling on the Criteo Uplift schema.

Default behavior:
    - Load a local Criteo Uplift Prediction Dataset file if present.
    - Otherwise generate same-schema semi-synthetic data.
    - Sample 500k rows by default.
    - Train Random, CVR Targeting, GMV Targeting, T-Learner and DR-Learner.
    - Use DR-Learner + LightGBM as the main model when LightGBM is installed.
    - Rank users by predicted incremental profit / coupon_cost under budget.

The Criteo public dataset has conversion labels but no user-level profit
counterfactuals. This script therefore adds a semi-synthetic profit layer on
top of the Criteo feature/treatment schema so that tau_profit, calibration,
budget sensitivity and Qini/AUUC can be evaluated end-to-end.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import struct
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


FEATURE_COLS = [f"f{i}" for i in range(12)]
CRITEO_REQUIRED_COLS = set(FEATURE_COLS + ["treatment", "conversion", "visit", "exposure"])
STRATEGIES = ["Random", "CVR Targeting", "GMV Targeting", "T-Learner", "DR-Learner"]
DATA_OUTPUT_COLS = (
    ["user_id"]
    + FEATURE_COLS
    + [
        "treatment",
        "conversion",
        "visit",
        "exposure",
        "coupon_cost",
        "gmv",
        "profit",
        "tau_profit",
        "source",
    ]
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    return math.log(p / (1 - p))


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 1e-5, 1 - 1e-5))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability)
        return np.column_stack([1 - p, p])


@dataclass
class ModelBundle:
    backend: str
    scores: pd.DataFrame
    nuisance: Dict[str, np.ndarray]
    diagnostics: Dict[str, float]
    artifacts: Dict[str, object]


def infer_backend(requested: str) -> str:
    if requested != "auto":
        if requested == "lightgbm":
            try:
                import lightgbm  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "LightGBM is not installed. Run `pip install -r requirements.txt`, "
                    "or use `--backend sklearn` for the lightweight backend."
                ) from exc
        return requested

    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        return "sklearn"


def make_regressor(backend: str, seed: int):
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=260,
            learning_rate=0.045,
            num_leaves=48,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=2.0, random_state=seed)),
        ]
    )


def make_classifier(backend: str, seed: int):
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=40,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_estimator(model, x: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
    if sample_weight is None:
        return model.fit(x, y)
    if isinstance(model, Pipeline):
        return model.fit(x, y, model__sample_weight=sample_weight)
    return model.fit(x, y, sample_weight=sample_weight)


def fit_classifier(backend: str, seed: int, x: pd.DataFrame, y: pd.Series):
    if y.nunique() < 2:
        return ConstantProbabilityModel(float(y.mean()))
    model = make_classifier(backend, seed)
    return fit_estimator(model, x, y)


def predict_proba_one(model, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    return np.asarray(proba[:, 1], dtype=float)


def find_local_criteo_file(search_roots: Iterable[Path]) -> Optional[Path]:
    def is_project_snapshot(path: Path) -> bool:
        return path.name.startswith("criteo_profit_uplift_sample")

    preferred_names = [
        "criteo-uplift-v2.1.csv",
        "criteo-uplift-v2.1.csv.gz",
        "criteo_uplift.csv",
        "criteo_uplift.csv.gz",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for name in preferred_names:
            candidate = root / name
            if candidate.exists():
                return candidate

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("*criteo*uplift*.csv", "*criteo*uplift*.csv.gz", "*criteo*.csv", "*criteo*.csv.gz"):
            matches = [path for path in sorted(root.rglob(pattern)) if not is_project_snapshot(path)]
            if matches:
                return matches[0]
    return None


def read_criteo(path: Path, sample_size: int) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, nrows=sample_size)
    except Exception as exc:
        warnings.warn(f"Failed to read {path}: {exc}. Falling back to semi-synthetic data.")
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = CRITEO_REQUIRED_COLS.difference(df.columns)
    if missing:
        warnings.warn(
            f"{path} is missing Criteo columns {sorted(missing)}. "
            "Falling back to same-schema semi-synthetic data."
        )
        return None
    ordered_cols = FEATURE_COLS + ["treatment", "conversion", "visit", "exposure"]
    return df[ordered_cols].copy()


def generate_criteo_like(sample_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(sample_size, 12))
    latent_value = rng.normal(size=sample_size)
    latent_activity = rng.normal(size=sample_size)

    features = pd.DataFrame(
        {
            "f0": z[:, 0] + 0.7 * latent_value,
            "f1": np.exp(0.35 * z[:, 1] + 0.25 * latent_activity),
            "f2": z[:, 2] - 0.2 * latent_value,
            "f3": np.maximum(0, 2.0 + z[:, 3] + latent_activity),
            "f4": z[:, 4] * z[:, 5],
            "f5": z[:, 5],
            "f6": np.exp(0.25 * z[:, 6]),
            "f7": z[:, 7] + 0.4 * latent_activity,
            "f8": z[:, 8],
            "f9": np.maximum(0, z[:, 9] + 1.5),
            "f10": z[:, 10] + 0.2 * latent_value,
            "f11": z[:, 11],
        }
    )
    treatment_propensity = sigmoid(-0.05 + 0.35 * features["f0"] - 0.20 * features["f2"] + 0.15 * features["f7"])
    features["treatment"] = rng.binomial(1, treatment_propensity)
    features["conversion"] = 0
    features["visit"] = 0
    features["exposure"] = features["treatment"]
    return features


def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = out[col].median()
        if pd.isna(median):
            median = 0.0
        out[col] = out[col].fillna(median)
    for col in ["treatment", "conversion", "visit", "exposure"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return out


def add_profit_layer(df: pd.DataFrame, seed: int, source: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 991)
    data = standardize_features(df)
    x = data[FEATURE_COLS].to_numpy(dtype=float)
    z = (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-6)

    price_sensitivity = sigmoid(0.90 * z[:, 6] - 0.35 * z[:, 0] + 0.30 * z[:, 9] + 0.15 * z[:, 11])
    organic_loyalty = sigmoid(0.75 * z[:, 0] + 0.30 * z[:, 3] - 0.25 * price_sensitivity)
    base_logit = -3.0 + 0.55 * z[:, 0] + 0.25 * z[:, 1] - 0.20 * z[:, 2] + 0.18 * z[:, 3]

    if "conversion" in df.columns and df["conversion"].mean() > 0:
        target_rate = float(np.clip(df["conversion"].mean(), 0.003, 0.20))
    else:
        target_rate = 0.045
    base_logit += logit(target_rate) - logit(float(sigmoid(base_logit).mean()))

    treatment_lift_logit = (
        -0.10
        + 1.25 * price_sensitivity
        + 0.18 * z[:, 8]
        - 0.75 * organic_loyalty * (1 - price_sensitivity)
        - 0.10 * np.maximum(z[:, 2], 0)
    )
    p0 = sigmoid(base_logit)
    p1 = sigmoid(base_logit + treatment_lift_logit)

    order_value = np.clip(18.0 + 4.0 * z[:, 0] + 3.5 * z[:, 3] + 2.0 * z[:, 10] + rng.normal(0, 2.0, len(df)), 4, 90)
    margin_rate = np.clip(0.22 + 0.04 * sigmoid(z[:, 5]) - 0.03 * price_sensitivity, 0.10, 0.42)
    margin0 = order_value * margin_rate
    basket_lift = np.clip(0.03 + 0.10 * price_sensitivity - 0.03 * organic_loyalty, -0.03, 0.18)
    margin1 = margin0 * (1.0 + basket_lift)
    coupon_cost = np.clip(0.45 + 0.45 * price_sensitivity + 0.18 * sigmoid(z[:, 10]) + rng.normal(0, 0.04, len(df)), 0.20, 2.20)

    treatment = data["treatment"].to_numpy(dtype=int)
    p_observed = np.where(treatment == 1, p1, p0)
    conversion = rng.binomial(1, p_observed)
    visit = np.maximum(conversion, rng.binomial(1, np.clip(2.8 * p_observed + 0.04, 0, 1)))
    exposure = np.where(treatment == 1, rng.binomial(1, np.clip(0.82 + 0.08 * sigmoid(z[:, 7]), 0, 1)), 0)

    observed_margin = np.where(treatment == 1, margin1, margin0)
    observed_order_value = np.where(treatment == 1, order_value * (1.0 + basket_lift), order_value)
    gmv = conversion * observed_order_value
    profit = conversion * (observed_margin - treatment * coupon_cost)

    data["user_id"] = [f"user_{i:07d}" for i in range(len(data))]
    data["conversion"] = conversion
    data["visit"] = visit
    data["exposure"] = exposure
    data["coupon_cost"] = coupon_cost
    data["gmv"] = gmv
    data["profit"] = profit
    data["tau_profit"] = p1 * (margin1 - coupon_cost) - p0 * margin0
    data["source"] = source
    return data


def load_dataset(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, str]]:
    metadata: Dict[str, str] = {}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.data_dir / f"criteo_profit_uplift_sample{args.sample_size}_seed{args.seed}.csv.gz"

    if not args.no_cache_data and not args.data_path and cache_path.exists():
        data = pd.read_csv(cache_path)
        metadata["data_source"] = str(cache_path)
        metadata["cache_used"] = "true"
        return data, metadata

    data_path = Path(args.data_path) if args.data_path else None
    if data_path and data_path.exists():
        raw = read_criteo(data_path, args.sample_size)
        if raw is not None:
            metadata["data_source"] = str(data_path)
            data = add_profit_layer(raw, args.seed, "local_criteo")
            if not args.no_save_data:
                data[DATA_OUTPUT_COLS].to_csv(cache_path, index=False, compression="gzip")
                metadata["saved_data_path"] = str(cache_path)
            return data, metadata

    search_roots = [Path.cwd(), Path.cwd() / "data", Path.cwd().parent / "data"]
    found = find_local_criteo_file(search_roots)
    if found is not None:
        raw = read_criteo(found, args.sample_size)
        if raw is not None:
            metadata["data_source"] = str(found)
            data = add_profit_layer(raw, args.seed, "local_criteo")
            if not args.no_save_data:
                data[DATA_OUTPUT_COLS].to_csv(cache_path, index=False, compression="gzip")
                metadata["saved_data_path"] = str(cache_path)
            return data, metadata

    metadata["data_source"] = "same_schema_semi_synthetic"
    raw = generate_criteo_like(args.sample_size, args.seed)
    data = add_profit_layer(raw, args.seed, "same_schema_semi_synthetic")
    if not args.no_save_data:
        data[DATA_OUTPUT_COLS].to_csv(cache_path, index=False, compression="gzip")
        metadata["saved_data_path"] = str(cache_path)
    return data, metadata


def fit_t_learner(train: pd.DataFrame, test: pd.DataFrame, backend: str, seed: int) -> Tuple[np.ndarray, Dict[str, object]]:
    x_train = train[FEATURE_COLS]
    x_test = test[FEATURE_COLS]
    treated = train["treatment"] == 1

    mu1 = make_regressor(backend, seed + 10)
    mu0 = make_regressor(backend, seed + 11)
    fit_estimator(mu1, x_train.loc[treated], train.loc[treated, "profit"])
    fit_estimator(mu0, x_train.loc[~treated], train.loc[~treated, "profit"])
    score = np.asarray(mu1.predict(x_test) - mu0.predict(x_test), dtype=float)
    return score, {"t_learner_mu1_model": mu1, "t_learner_mu0_model": mu0}


def fit_dr_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    backend: str,
    seed: int,
    n_folds: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], object]:
    x = train[FEATURE_COLS].reset_index(drop=True)
    y = train["profit"].reset_index(drop=True)
    w = train["treatment"].reset_index(drop=True)
    x_test = test[FEATURE_COLS]

    min_arm_size = int(min(w.sum(), len(w) - w.sum()))
    actual_folds = max(2, min(n_folds, min_arm_size))
    kfold = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    mu0_oof = np.zeros(len(train))
    mu1_oof = np.zeros(len(train))
    e_oof = np.zeros(len(train))

    for fold, (fit_idx, pred_idx) in enumerate(kfold.split(x, w)):
        fit_x = x.iloc[fit_idx]
        fit_y = y.iloc[fit_idx]
        fit_w = w.iloc[fit_idx]
        pred_x = x.iloc[pred_idx]

        mu1 = make_regressor(backend, seed + 100 + fold)
        mu0 = make_regressor(backend, seed + 200 + fold)
        fit_estimator(mu1, fit_x.loc[fit_w == 1], fit_y.loc[fit_w == 1])
        fit_estimator(mu0, fit_x.loc[fit_w == 0], fit_y.loc[fit_w == 0])
        propensity = fit_classifier(backend, seed + 300 + fold, fit_x, fit_w)

        mu1_oof[pred_idx] = mu1.predict(pred_x)
        mu0_oof[pred_idx] = mu0.predict(pred_x)
        e_oof[pred_idx] = predict_proba_one(propensity, pred_x)

    e_oof = np.clip(e_oof, 0.02, 0.98)
    pseudo_tau = (
        mu1_oof
        - mu0_oof
        + w.to_numpy() * (y.to_numpy() - mu1_oof) / e_oof
        - (1 - w.to_numpy()) * (y.to_numpy() - mu0_oof) / (1 - e_oof)
    )

    tau_model = make_regressor(backend, seed + 400)
    fit_estimator(tau_model, x, pd.Series(pseudo_tau))

    mu1_full = make_regressor(backend, seed + 500)
    mu0_full = make_regressor(backend, seed + 501)
    propensity_full = fit_classifier(backend, seed + 502, x, w)
    fit_estimator(mu1_full, x.loc[w == 1], y.loc[w == 1])
    fit_estimator(mu0_full, x.loc[w == 0], y.loc[w == 0])

    test_mu1 = np.asarray(mu1_full.predict(x_test), dtype=float)
    test_mu0 = np.asarray(mu0_full.predict(x_test), dtype=float)
    test_e = np.clip(predict_proba_one(propensity_full, x_test), 0.02, 0.98)
    test_w = test["treatment"].to_numpy()
    test_y = test["profit"].to_numpy()
    test_dr_pseudo = (
        test_mu1
        - test_mu0
        + test_w * (test_y - test_mu1) / test_e
        - (1 - test_w) * (test_y - test_mu0) / (1 - test_e)
    )

    nuisance = {
        "mu1": test_mu1,
        "mu0": test_mu0,
        "propensity": test_e,
        "dr_pseudo_tau": test_dr_pseudo,
    }
    artifacts = {
        "dr_tau_model": tau_model,
        "dr_mu1_model": mu1_full,
        "dr_mu0_model": mu0_full,
        "dr_propensity_model": propensity_full,
    }
    return np.asarray(tau_model.predict(x_test), dtype=float), nuisance, artifacts


def fit_scores(train: pd.DataFrame, test: pd.DataFrame, backend: str, seed: int, n_folds: int) -> ModelBundle:
    x_train = train[FEATURE_COLS]
    x_test = test[FEATURE_COLS]
    rng = np.random.default_rng(seed + 202)

    scores = pd.DataFrame(index=test.index)
    scores["Random"] = rng.random(len(test))

    cvr_model = fit_classifier(backend, seed + 1, x_train, train["conversion"])
    scores["CVR Targeting"] = predict_proba_one(cvr_model, x_test)

    gmv_model = make_regressor(backend, seed + 2)
    fit_estimator(gmv_model, x_train, train["gmv"])
    scores["GMV Targeting"] = np.asarray(gmv_model.predict(x_test), dtype=float)

    t_score, t_artifacts = fit_t_learner(train, test, backend, seed)
    scores["T-Learner"] = t_score
    dr_score, nuisance, dr_artifacts = fit_dr_learner(train, test, backend, seed, n_folds)
    scores["DR-Learner"] = dr_score

    diagnostics: Dict[str, float] = {}
    try:
        diagnostics["conversion_auc"] = float(roc_auc_score(test["conversion"], scores["CVR Targeting"]))
    except ValueError:
        diagnostics["conversion_auc"] = float("nan")
    diagnostics["dr_tau_corr"] = float(np.corrcoef(scores["DR-Learner"], test["tau_profit"])[0, 1])
    diagnostics["t_tau_corr"] = float(np.corrcoef(scores["T-Learner"], test["tau_profit"])[0, 1])
    diagnostics["crossfit_folds"] = float(n_folds)
    artifacts = {
        "cvr_model": cvr_model,
        "gmv_model": gmv_model,
        **t_artifacts,
        **dr_artifacts,
    }
    return ModelBundle(backend=backend, scores=scores, nuisance=nuisance, diagnostics=diagnostics, artifacts=artifacts)


def allocate_by_ratio(score: np.ndarray, coupon_cost: np.ndarray, budget: float, positive_only: bool) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    coupon_cost = np.asarray(coupon_cost, dtype=float)
    ratio = score / np.clip(coupon_cost, 1e-6, None)
    order = np.argsort(-ratio)
    selected = np.zeros(len(score), dtype=bool)
    spent = 0.0
    for idx in order:
        if positive_only and score[idx] <= 0:
            continue
        cost = coupon_cost[idx]
        if not np.isfinite(ratio[idx]) or cost <= 0:
            continue
        if spent + cost > budget:
            continue
        selected[idx] = True
        spent += cost
    return selected


def qini_curve(score: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_points: int = 200) -> pd.DataFrame:
    order = np.argsort(-np.asarray(score, dtype=float))
    w = np.asarray(treatment, dtype=int)[order]
    y = np.asarray(outcome, dtype=float)[order]
    n = len(w)

    cum_t = np.cumsum(w)
    cum_c = np.cumsum(1 - w)
    cum_y_t = np.cumsum(y * w)
    cum_y_c = np.cumsum(y * (1 - w))

    points = np.unique(np.linspace(1, n, min(n_points, n), dtype=int)) - 1
    rows = []
    for idx in points:
        nt = max(cum_t[idx], 1)
        nc = max(cum_c[idx], 1)
        population = idx + 1
        qini = cum_y_t[idx] - cum_y_c[idx] * nt / nc
        uplift_gain = (cum_y_t[idx] / nt - cum_y_c[idx] / nc) * population
        rows.append(
            {
                "population_fraction": population / n,
                "qini_gain": float(qini),
                "uplift_gain": float(uplift_gain),
            }
        )
    return pd.DataFrame(rows)


def curve_metrics(curve: pd.DataFrame) -> Tuple[float, float]:
    x = curve["population_fraction"].to_numpy()
    uplift_y = curve["uplift_gain"].to_numpy()
    qini_y = curve["qini_gain"].to_numpy()
    auuc_value = float(auc(x, uplift_y))
    random_line = np.linspace(0.0, qini_y[-1], len(qini_y))
    qini_value = float(auc(x, qini_y - random_line))
    return auuc_value, qini_value


def bootstrap_sum_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if n_bootstrap <= 0 or len(values) == 0:
        total = float(values.sum())
        return total, total
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap)
    n = len(values)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        estimates[i] = values[idx].sum()
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def oracle_tau_auuc(score: np.ndarray, tau_profit: np.ndarray, n_points: int = 200) -> float:
    order = np.argsort(-np.asarray(score, dtype=float))
    tau = np.asarray(tau_profit, dtype=float)[order]
    n = len(tau)
    points = np.unique(np.linspace(1, n, min(n_points, n), dtype=int))
    x = points / n
    y = np.array([tau[:point].sum() for point in points], dtype=float)
    return float(auc(x, y))


def evaluate_strategy(
    name: str,
    score: np.ndarray,
    test: pd.DataFrame,
    budget: float,
    dr_pseudo_tau: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[dict, np.ndarray, pd.DataFrame]:
    positive_only = name in {"T-Learner", "DR-Learner"}
    selected = allocate_by_ratio(score, test["coupon_cost"].to_numpy(), budget, positive_only)
    incremental_profit = float(test.loc[selected, "tau_profit"].sum())
    total_cost = float(test.loc[selected, "coupon_cost"].sum())
    aipw_values = selected.astype(float) * np.asarray(dr_pseudo_tau, dtype=float)
    aipw_incremental_profit = float(aipw_values.sum())
    ci_low, ci_high = bootstrap_sum_ci(aipw_values, n_bootstrap, seed)
    selected_users = int(selected.sum())
    curve = qini_curve(score, test["treatment"].to_numpy(), test["profit"].to_numpy())
    auuc_value, qini_value = curve_metrics(curve)
    row = {
        "strategy": name,
        "selected_users": selected_users,
        "selected_rate": selected_users / len(test),
        "coupon_cost": total_cost,
        "incremental_profit": incremental_profit,
        "roi": incremental_profit / total_cost if total_cost > 0 else 0.0,
        "aipw_incremental_profit": aipw_incremental_profit,
        "aipw_roi": aipw_incremental_profit / total_cost if total_cost > 0 else 0.0,
        "aipw_ci_low": ci_low,
        "aipw_ci_high": ci_high,
        "auuc": auuc_value,
        "qini": qini_value,
        "oracle_tau_auuc": oracle_tau_auuc(score, test["tau_profit"].to_numpy()),
        "avg_predicted_incremental_profit": float(np.mean(score[selected])) if selected_users else 0.0,
        "avg_true_tau_profit": float(test.loc[selected, "tau_profit"].mean()) if selected_users else 0.0,
    }
    return row, selected, curve


def budget_sensitivity(scores: pd.DataFrame, test: pd.DataFrame, max_budget: float, dr_pseudo_tau: np.ndarray) -> pd.DataFrame:
    budgets = np.linspace(0.1, 1.0, 10) * max_budget
    rows = []
    for budget in budgets:
        for strategy in STRATEGIES:
            selected = allocate_by_ratio(
                scores[strategy].to_numpy(),
                test["coupon_cost"].to_numpy(),
                budget,
                strategy in {"T-Learner", "DR-Learner"},
            )
            cost = float(test.loc[selected, "coupon_cost"].sum())
            inc = float(test.loc[selected, "tau_profit"].sum())
            aipw_inc = float((selected.astype(float) * dr_pseudo_tau).sum())
            rows.append(
                {
                    "strategy": strategy,
                    "budget": float(budget),
                    "selected_users": int(selected.sum()),
                    "coupon_cost": cost,
                    "incremental_profit": inc,
                    "roi": inc / cost if cost > 0 else 0.0,
                    "aipw_incremental_profit": aipw_inc,
                    "aipw_roi": aipw_inc / cost if cost > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def uplift_decile_calibration(test: pd.DataFrame, dr_score: np.ndarray, dr_pseudo: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "predicted_tau_profit": dr_score,
            "true_tau_profit": test["tau_profit"].to_numpy(),
            "dr_observed_tau_estimate": dr_pseudo,
            "coupon_cost": test["coupon_cost"].to_numpy(),
            "selected": selected.astype(int),
        }
    )
    order = frame["predicted_tau_profit"].rank(method="first", ascending=False)
    frame["decile"] = pd.qcut(order, 10, labels=False) + 1
    rows = []
    for decile, part in frame.groupby("decile", sort=True):
        rows.append(
            {
                "decile": int(decile),
                "n_users": int(len(part)),
                "avg_predicted_tau_profit": float(part["predicted_tau_profit"].mean()),
                "avg_true_tau_profit": float(part["true_tau_profit"].mean()),
                "avg_dr_observed_tau_estimate": float(part["dr_observed_tau_estimate"].mean()),
                "calibration_gap": float(part["predicted_tau_profit"].mean() - part["true_tau_profit"].mean()),
                "avg_coupon_cost": float(part["coupon_cost"].mean()),
                "selected_rate": float(part["selected"].mean()),
            }
        )
    return pd.DataFrame(rows)


def add_segment_row(rows: List[dict], name: str, mask: np.ndarray, test: pd.DataFrame, dr_score: np.ndarray, selected: np.ndarray) -> None:
    if mask.sum() == 0:
        return
    segment_selected = mask & selected
    cost = float(test.loc[segment_selected, "coupon_cost"].sum())
    inc = float(test.loc[segment_selected, "tau_profit"].sum())
    rows.append(
        {
            "segment": name,
            "n_users": int(mask.sum()),
            "selected_users": int(segment_selected.sum()),
            "selected_rate": float(segment_selected.sum() / mask.sum()),
            "avg_predicted_tau_profit": float(np.mean(dr_score[mask])),
            "avg_true_tau_profit": float(test.loc[mask, "tau_profit"].mean()),
            "coupon_cost": cost,
            "incremental_profit": inc,
            "roi": inc / cost if cost > 0 else 0.0,
        }
    )


def segment_diagnostics(test: pd.DataFrame, dr_score: np.ndarray, propensity: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    rows: List[dict] = []
    n = len(test)
    all_mask = np.ones(n, dtype=bool)
    add_segment_row(rows, "all", all_mask, test, dr_score, selected)

    cost = test["coupon_cost"].to_numpy()
    low_cost, high_cost = np.quantile(cost, [0.33, 0.67])
    add_segment_row(rows, "coupon_cost_low", cost <= low_cost, test, dr_score, selected)
    add_segment_row(rows, "coupon_cost_mid", (cost > low_cost) & (cost <= high_cost), test, dr_score, selected)
    add_segment_row(rows, "coupon_cost_high", cost > high_cost, test, dr_score, selected)

    low_prop, high_prop = np.quantile(propensity, [0.33, 0.67])
    add_segment_row(rows, "propensity_low", propensity <= low_prop, test, dr_score, selected)
    add_segment_row(rows, "propensity_mid", (propensity > low_prop) & (propensity <= high_prop), test, dr_score, selected)
    add_segment_row(rows, "propensity_high", propensity > high_prop, test, dr_score, selected)

    for feature in ["f0", "f6", "f10"]:
        values = test[feature].to_numpy()
        threshold = np.median(values)
        add_segment_row(rows, f"{feature}_low", values <= threshold, test, dr_score, selected)
        add_segment_row(rows, f"{feature}_high", values > threshold, test, dr_score, selected)

    return pd.DataFrame(rows)


def overlap_diagnostics(test: pd.DataFrame, propensity: np.ndarray, selected: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "propensity": propensity,
            "treatment": test["treatment"].to_numpy(),
            "tau_profit": test["tau_profit"].to_numpy(),
            "coupon_cost": test["coupon_cost"].to_numpy(),
            "selected": selected.astype(int),
        }
    )
    order = frame["propensity"].rank(method="first")
    frame["propensity_decile"] = pd.qcut(order, 10, labels=False) + 1
    rows = []
    for decile, part in frame.groupby("propensity_decile", sort=True):
        rows.append(
            {
                "propensity_decile": int(decile),
                "n_users": int(len(part)),
                "avg_propensity": float(part["propensity"].mean()),
                "min_propensity": float(part["propensity"].min()),
                "max_propensity": float(part["propensity"].max()),
                "observed_treatment_rate": float(part["treatment"].mean()),
                "selected_rate": float(part["selected"].mean()),
                "avg_tau_profit": float(part["tau_profit"].mean()),
                "avg_coupon_cost": float(part["coupon_cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


def model_feature_diagnostics(bundle: ModelBundle, test: pd.DataFrame, selected: np.ndarray) -> pd.DataFrame:
    model = bundle.artifacts.get("dr_tau_model")
    model_importance = np.full(len(FEATURE_COLS), np.nan)

    if hasattr(model, "feature_importances_"):
        model_importance = np.asarray(model.feature_importances_, dtype=float)
    elif isinstance(model, Pipeline) and "model" in model.named_steps and hasattr(model.named_steps["model"], "coef_"):
        coef = np.asarray(model.named_steps["model"].coef_, dtype=float).reshape(-1)
        if len(coef) == len(FEATURE_COLS):
            model_importance = np.abs(coef)

    dr_score = bundle.scores["DR-Learner"].to_numpy()
    true_tau = test["tau_profit"].to_numpy()
    rows = []
    for idx, feature in enumerate(FEATURE_COLS):
        values = test[feature].to_numpy(dtype=float)
        corr_pred = np.corrcoef(values, dr_score)[0, 1]
        corr_true = np.corrcoef(values, true_tau)[0, 1]
        selected_mean = float(np.mean(values[selected])) if selected.any() else float("nan")
        non_selected_mean = float(np.mean(values[~selected])) if (~selected).any() else float("nan")
        rows.append(
            {
                "feature": feature,
                "model_importance": float(model_importance[idx]) if np.isfinite(model_importance[idx]) else np.nan,
                "corr_with_predicted_tau": float(corr_pred),
                "corr_with_true_tau": float(corr_true),
                "selected_mean": selected_mean,
                "non_selected_mean": non_selected_mean,
                "selected_minus_non_selected": selected_mean - non_selected_mean,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_importance", "corr_with_predicted_tau"], ascending=False)


def write_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], width: int = 2) -> None:
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    radius = max(0, width // 2)
    h, w, _ = image.shape
    for x, y in zip(xs, ys):
        x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
        image[y_min:y_max, x_min:x_max] = color


def draw_rect(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    h, w, _ = image.shape
    x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
    image[y0 : y1 + 1, x0 : x1 + 1] = color


def scale_points(x: np.ndarray, y: np.ndarray, width: int, height: int, margin: int, y_min: float, y_max: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.nanmax(x) == np.nanmin(x):
        xs = np.full(len(x), margin)
    else:
        xs = margin + (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) * (width - 2 * margin)
    if y_max == y_min:
        ys = np.full(len(y), height // 2)
    else:
        ys = height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)
    return xs.astype(int), ys.astype(int)


def save_simple_line_png(output_path: Path, series: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    width, height, margin = 900, 560, 56
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    axis_color = (70, 70, 70)
    draw_line(image, margin, height - margin, width - margin, height - margin, axis_color, 2)
    draw_line(image, margin, margin, margin, height - margin, axis_color, 2)

    colors = [
        (76, 120, 168),
        (245, 133, 24),
        (84, 162, 75),
        (182, 70, 72),
        (114, 78, 145),
        (72, 170, 173),
    ]
    all_y = np.concatenate([np.asarray(y, dtype=float) for _, y in series.values()])
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    pad = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= pad
    y_max += pad

    for idx, (_, (x, y)) in enumerate(series.items()):
        xs, ys = scale_points(x, y, width, height, margin, y_min, y_max)
        color = colors[idx % len(colors)]
        for i in range(1, len(xs)):
            draw_line(image, int(xs[i - 1]), int(ys[i - 1]), int(xs[i]), int(ys[i]), color, 3)
        draw_rect(image, width - margin - 22, margin + idx * 18, width - margin - 6, margin + idx * 18 + 10, color)
    write_png(output_path, image)


def save_simple_bar_png(output_path: Path, values: np.ndarray) -> None:
    width, height, margin = 900, 560, 56
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    axis_color = (70, 70, 70)
    draw_line(image, margin, height - margin, width - margin, height - margin, axis_color, 2)
    draw_line(image, margin, margin, margin, height - margin, axis_color, 2)

    values = np.asarray(values, dtype=float)
    y_min = min(0.0, float(np.nanmin(values)))
    y_max = max(0.0, float(np.nanmax(values)))
    pad = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= pad
    y_max += pad
    bar_width = max(12, int((width - 2 * margin) / max(len(values), 1) * 0.65))
    colors = [(76, 120, 168), (245, 133, 24), (84, 162, 75), (182, 70, 72), (114, 78, 145)]

    zero_y = int(height - margin - (0 - y_min) / (y_max - y_min) * (height - 2 * margin))
    for idx, value in enumerate(values):
        center = int(margin + (idx + 0.5) * (width - 2 * margin) / len(values))
        top = int(height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin))
        draw_rect(image, center - bar_width // 2, min(top, zero_y), center + bar_width // 2, max(top, zero_y), colors[idx % len(colors)])
    write_png(output_path, image)


def save_qini_plot(curves: Dict[str, pd.DataFrame], output_path: Path) -> None:
    if plt is None:
        series = {
            strategy: (curve["population_fraction"].to_numpy(), curve["qini_gain"].to_numpy())
            for strategy, curve in curves.items()
        }
        save_simple_line_png(output_path, series)
        return

    plt.figure(figsize=(8, 5))
    for strategy, curve in curves.items():
        plt.plot(curve["population_fraction"], curve["qini_gain"], label=strategy, linewidth=1.8)
    plt.axhline(0, color="#777777", linewidth=0.8)
    plt.xlabel("Population fraction ranked by score")
    plt.ylabel("Qini gain (observed profit)")
    plt.title("Qini Curve")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_budget_plot(budget_curve: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        series = {
            strategy: (part["budget"].to_numpy(), part["incremental_profit"].to_numpy())
            for strategy, part in budget_curve.groupby("strategy")
        }
        save_simple_line_png(output_path, series)
        return

    plt.figure(figsize=(8, 5))
    for strategy, part in budget_curve.groupby("strategy"):
        plt.plot(part["budget"], part["incremental_profit"], marker="o", label=strategy, linewidth=1.8)
    plt.xlabel("Budget")
    plt.ylabel("Incremental profit")
    plt.title("Budget Sensitivity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_calibration_plot(calibration: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        series = {
            "Predicted": (
                calibration["decile"].to_numpy(),
                calibration["avg_predicted_tau_profit"].to_numpy(),
            ),
            "True": (
                calibration["decile"].to_numpy(),
                calibration["avg_true_tau_profit"].to_numpy(),
            ),
        }
        save_simple_line_png(output_path, series)
        return

    plt.figure(figsize=(8, 5))
    plt.plot(calibration["decile"], calibration["avg_predicted_tau_profit"], marker="o", label="Predicted")
    plt.plot(calibration["decile"], calibration["avg_true_tau_profit"], marker="s", label="True")
    plt.gca().invert_xaxis()
    plt.xlabel("Predicted uplift decile (1 = highest)")
    plt.ylabel("Average tau_profit")
    plt.title("Uplift Decile Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_strategy_plot(strategy_comparison: pd.DataFrame, output_path: Path) -> None:
    if plt is None:
        values = strategy_comparison["incremental_profit"].to_numpy()
        save_simple_bar_png(output_path, values)
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(strategy_comparison))
    ax1.bar(x, strategy_comparison["incremental_profit"], color="#4C78A8", alpha=0.82, label="Incremental profit")
    ax1.set_ylabel("Incremental profit")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_comparison["strategy"], rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, strategy_comparison["roi"], color="#F58518", marker="o", label="ROI")
    ax2.set_ylabel("ROI")
    fig.suptitle("Strategy Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_model_artifacts(
    model_dir: Path,
    bundle: ModelBundle,
    metadata: Dict[str, str],
    config: Dict[str, object],
) -> Dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_files: Dict[str, str] = {}

    for name, model in bundle.artifacts.items():
        path = model_dir / f"{name}.pkl"
        payload = {
            "model_name": name,
            "backend": bundle.backend,
            "feature_cols": FEATURE_COLS,
            "model": model,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        model_files[name] = str(path)

    model_card = {
        "main_model": "DR-Learner + LightGBM" if bundle.backend == "lightgbm" else "DR-Learner + sklearn backend",
        "backend": bundle.backend,
        "feature_cols": FEATURE_COLS,
        "target": "tau_profit = E[profit|treatment=1,x] - E[profit|treatment=0,x]",
        "ranking_rule": "predicted_tau_profit / coupon_cost",
        "data": metadata,
        "config": config,
        "models": model_files,
        "load_example": "payload = pickle.load(open('models/dr_tau_model.pkl', 'rb')); model = payload['model']; model.predict(df[payload['feature_cols']])",
        "notes": [
            "dr_tau_model.pkl is the production scoring model for predicted incremental profit.",
            "dr_mu1_model.pkl, dr_mu0_model.pkl and dr_propensity_model.pkl are nuisance models for audit/AIPW evaluation.",
            "t_learner_mu1_model.pkl and t_learner_mu0_model.pkl reproduce the T-Learner baseline.",
        ],
    }
    card_path = model_dir / "model_card.json"
    with card_path.open("w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    return {"model_dir": str(model_dir), "model_card": str(card_path)}


def write_outputs(
    output_dir: Path,
    bundle: ModelBundle,
    test: pd.DataFrame,
    budget: float,
    metadata: Dict[str, str],
    n_bootstrap: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_rows = []
    selected_by_strategy: Dict[str, np.ndarray] = {}
    curves: Dict[str, pd.DataFrame] = {}

    for strategy in STRATEGIES:
        row, selected, curve = evaluate_strategy(
            strategy,
            bundle.scores[strategy].to_numpy(),
            test,
            budget,
            bundle.nuisance["dr_pseudo_tau"],
            n_bootstrap,
            seed + 1000 + len(strategy),
        )
        strategy_rows.append(row)
        selected_by_strategy[strategy] = selected
        curves[strategy] = curve

    strategy_comparison = pd.DataFrame(strategy_rows).sort_values("incremental_profit", ascending=False)
    budget_curve = budget_sensitivity(bundle.scores, test, budget, bundle.nuisance["dr_pseudo_tau"])
    calibration = uplift_decile_calibration(
        test,
        bundle.scores["DR-Learner"].to_numpy(),
        bundle.nuisance["dr_pseudo_tau"],
        selected_by_strategy["DR-Learner"],
    )
    segments = segment_diagnostics(
        test,
        bundle.scores["DR-Learner"].to_numpy(),
        bundle.nuisance["propensity"],
        selected_by_strategy["DR-Learner"],
    )
    overlap = overlap_diagnostics(test, bundle.nuisance["propensity"], selected_by_strategy["DR-Learner"])
    feature_importance = model_feature_diagnostics(bundle, test, selected_by_strategy["DR-Learner"])

    strategy_comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
    budget_curve.to_csv(output_dir / "budget_curve.csv", index=False)
    calibration.to_csv(output_dir / "uplift_decile_calibration.csv", index=False)
    segments.to_csv(output_dir / "segment_analysis.csv", index=False)
    overlap.to_csv(output_dir / "overlap_diagnostics.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    save_qini_plot(curves, output_dir / "qini_curve.png")
    save_budget_plot(budget_curve, output_dir / "budget_profit_curve.png")
    save_calibration_plot(calibration, output_dir / "uplift_calibration.png")
    save_strategy_plot(strategy_comparison, output_dir / "strategy_comparison.png")

    run_metadata = {
        **metadata,
        "backend": bundle.backend,
        "budget": budget,
        "n_test": int(len(test)),
        "n_bootstrap": int(n_bootstrap),
        "diagnostics": bundle.diagnostics,
        "main_model": "DR-Learner + LightGBM" if bundle.backend == "lightgbm" else "DR-Learner + sklearn backend",
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)

    print("\nStrategy comparison")
    print(strategy_comparison.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nOutput directory: {output_dir}")
    print(f"Backend: {bundle.backend}")
    print(f"Data source: {metadata.get('data_source')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="", help="Optional local Criteo uplift CSV/CSV.GZ path.")
    parser.add_argument("--sample-size", type=int, default=500_000, help="Default sample size is 500k rows.")
    parser.add_argument("--budget", type=float, default=25_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["auto", "lightgbm", "sklearn"], default="sklearn")
    parser.add_argument("--n-folds", type=int, default=3, help="Cross-fitting folds for the DR-Learner.")
    parser.add_argument("--n-bootstrap", type=int, default=50, help="Bootstrap repeats for AIPW policy value CI.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for the reproducible data snapshot.")
    parser.add_argument("--no-cache-data", action="store_true", help="Ignore saved data snapshot and rebuild from source.")
    parser.add_argument("--no-save-data", action="store_true", help="Do not write the processed data snapshot.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for trained model artifacts.")
    parser.add_argument("--no-save-model", action="store_true", help="Do not write trained model artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    backend = infer_backend(args.backend)
    data, metadata = load_dataset(args)
    train, test = train_test_split(
        data,
        test_size=0.35,
        random_state=args.seed,
        stratify=data["treatment"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    bundle = fit_scores(train, test, backend, args.seed, args.n_folds)
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["backend_resolved"] = backend
    if not args.no_save_model:
        metadata.update(save_model_artifacts(args.model_dir, bundle, metadata, config))
    write_outputs(args.output_dir, bundle, test, args.budget, metadata, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
