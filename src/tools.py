"""Phase 2: the tools the agents use.

One ExperimentSession = one agent run on one dataset.
It holds the train/validation data, the experiment budget, and the MLflow
experiment. The test set is NOT loaded here; only final_test() reads it, once.

Tools (what the agents can call):
    profile_data()        -> summary of the training data
    train_and_evaluate()  -> train one model, score on validation, log to MLflow
    get_leaderboard()     -> read past experiments back from MLflow, best first
    final_test()          -> retrain best config on train+val, score on test ONCE
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACTS = ROOT / "mlartifacts"
SESSIONS = ROOT / "data" / "sessions"

# ---------------------------------------------------------------------------
# Models the agent may choose, and the settings it is allowed to change.
# Anything not listed here is rejected, so the agent can't pass nonsense.
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "logistic_regression": {
        "description": "Linear baseline (scikit-learn). Fast, interpretable.",
        "defaults": {"C": 1.0, "class_weight": None, "max_iter": 2000},
        "allowed": {"C", "class_weight", "max_iter"},
    },
    "random_forest": {
        "description": "Bagged decision trees (scikit-learn).",
        "defaults": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
                     "max_features": "sqrt", "class_weight": None},
        "allowed": {"n_estimators", "max_depth", "min_samples_leaf",
                    "max_features", "class_weight"},
    },
    "lightgbm": {
        "description": "Gradient-boosted trees (LightGBM). Usually strongest on tables.",
        "defaults": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                     "max_depth": -1, "min_child_samples": 20, "subsample": 1.0,
                     "colsample_bytree": 1.0, "reg_lambda": 0.0, "class_weight": None},
        "allowed": {"n_estimators", "learning_rate", "num_leaves", "max_depth",
                    "min_child_samples", "subsample", "colsample_bytree",
                    "reg_lambda", "class_weight"},
    },
    "mlp": {
        "description": "Small neural network (scikit-learn MLPClassifier).",
        "defaults": {"hidden_layer_sizes": [64, 32], "alpha": 0.0001,
                     "learning_rate_init": 0.001, "max_iter": 200},
        "allowed": {"hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter"},
    },
}


def _build_model(model_type: str, params: dict):
    p = dict(params)
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=SEED, **p)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **p)
    if model_type == "lightgbm":
        if p.get("subsample", 1.0) < 1.0:
            p["subsample_freq"] = 1  # LightGBM only subsamples rows when this is set
        return LGBMClassifier(random_state=SEED, verbose=-1, **p)
    if model_type == "mlp":
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPClassifier(random_state=SEED, early_stopping=True, **p)
    raise ValueError(model_type)


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numbers: fill gaps with the median, then scale (needed by logreg and MLP).
    Text/category columns: fill gaps with the most common value, then one-hot."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
    }


class ExperimentSession:
    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, test_path: Path,
                 target: str, budget: int = 15, name: str | None = None):
        if target not in train.columns:
            raise ValueError(f"Target column '{target}' not found.")
        self.train, self.val = train, val
        self._test_path = Path(test_path)   # only a path; the data is not loaded
        self._test_used = False
        self.target = target
        self.budget = budget
        self.used = 0
        self.history: dict[str, dict] = {}   # run_id -> {model_type, params, features}

        mlflow.set_tracking_uri(TRACKING_URI)
        self.name = name or f"session-{datetime.now():%Y%m%d-%H%M%S}"
        exp = mlflow.get_experiment_by_name(self.name)
        self.experiment_id = exp.experiment_id if exp else mlflow.create_experiment(
            self.name, artifact_location=(ARTIFACTS / self.name).as_uri())

    # ---------- ways to start a session ----------
    @classmethod
    def from_prepared_splits(cls, budget: int = 15, name: str | None = None):
        """Use the credit card default splits made by prepare_data.py."""
        d = ROOT / "data"
        return cls(pd.read_csv(d / "splits" / "train.csv"),
                   pd.read_csv(d / "splits" / "val.csv"),
                   d / "locked" / "test.csv", target="default",
                   budget=budget, name=name)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target: str, budget: int = 15,
                       name: str | None = None):
        """For any uploaded CSV: split 60/20/20 and lock the test part on disk."""
        df = df.dropna(subset=[target]).copy()
        if df[target].nunique() != 2:
            raise ValueError("Target must have exactly 2 classes (binary classification).")
        classes = sorted(df[target].unique(), key=str)
        df[target] = (df[target] == classes[1]).astype(int)

        train_val, test = train_test_split(df, test_size=0.2, stratify=df[target],
                                           random_state=SEED)
        train, val = train_test_split(train_val, test_size=0.25,
                                      stratify=train_val[target], random_state=SEED)
        locked = SESSIONS / uuid.uuid4().hex[:8]
        locked.mkdir(parents=True, exist_ok=True)
        test.to_csv(locked / "test.csv", index=False)
        return cls(train.reset_index(drop=True), val.reset_index(drop=True),
                   locked / "test.csv", target=target, budget=budget, name=name)

    @property
    def features(self) -> list[str]:
        return [c for c in self.train.columns if c != self.target]

    # ======================= TOOL 1 =======================
    def profile_data(self) -> dict:
        """Summary of the TRAINING data only."""
        X, y = self.train[self.features], self.train[self.target]
        num = X.select_dtypes(include="number")
        corr = num.corrwith(y).abs().sort_values(ascending=False)
        return {
            "rows_train": len(self.train),
            "rows_validation": len(self.val),
            "n_features": X.shape[1],
            "target": self.target,
            "positive_rate": round(float(y.mean()), 4),
            "numeric_features": num.columns.tolist(),
            "categorical_features": [c for c in X.columns if c not in num.columns],
            "missing_values": {c: int(v) for c, v in X.isna().sum().items() if v > 0},
            "top_correlations_with_target": {c: round(float(v), 3)
                                             for c, v in corr.head(8).items()},
            "models_available": {k: v["description"] for k, v in MODEL_SPECS.items()},
            "budget": self.budget,
        }

    # ======================= TOOL 2 =======================
    def train_and_evaluate(self, model_type: str, params: dict | None = None,
                           features: list[str] | None = None, notes: str = "") -> dict:
        """Train one model on train, score on validation, log to MLflow."""
        if self.used >= self.budget:
            return {"error": "Budget exhausted. No more experiments allowed."}
        if model_type not in MODEL_SPECS:
            return {"error": f"Unknown model_type. Choose from {list(MODEL_SPECS)}."}

        spec = MODEL_SPECS[model_type]
        params = params or {}
        bad = set(params) - spec["allowed"]
        if bad:
            return {"error": f"Not allowed for {model_type}: {sorted(bad)}. "
                             f"Allowed: {sorted(spec['allowed'])}."}
        full_params = {**spec["defaults"], **params}

        features = features or self.features
        unknown = [f for f in features if f not in self.features]
        if unknown:
            return {"error": f"Unknown features: {unknown}."}

        X_tr, y_tr = self.train[features], self.train[self.target]
        X_va, y_va = self.val[features], self.val[self.target]

        try:
            pipe = Pipeline([("prep", _preprocessor(X_tr)),
                             ("model", _build_model(model_type, full_params))])
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            fit_seconds = round(time.time() - t0, 2)
        except Exception as e:  # bad value (e.g. negative learning rate)
            return {"error": f"Training failed: {e}"}

        self.used += 1
        val_m = _metrics(y_va, pipe.predict_proba(X_va)[:, 1])
        train_auc = round(float(roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])), 4)

        with mlflow.start_run(experiment_id=self.experiment_id,
                              run_name=f"{self.used:02d}-{model_type}") as run:
            mlflow.set_tags({"model_type": model_type, "notes": notes[:500],
                             "experiment_number": self.used,
                             "n_features": len(features)})
            mlflow.log_params({k: str(v) for k, v in full_params.items()})
            mlflow.log_dict({"features": features}, "features.json")
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
            mlflow.log_metrics({"train_auc": train_auc, "fit_seconds": fit_seconds})
            run_id = run.info.run_id

        self.history[run_id] = {"model_type": model_type, "params": full_params,
                                "features": features}
        return {
            "run_id": run_id,
            "experiment_number": self.used,
            "model_type": model_type,
            "params": full_params,
            "n_features": len(features),
            "val_metrics": val_m,
            "train_auc": train_auc,
            "overfit_gap": round(train_auc - val_m["auc"], 4),
            "fit_seconds": fit_seconds,
            "budget_remaining": self.budget - self.used,
        }

    # ======================= TOOL 3 =======================
    def get_leaderboard(self, top_n: int = 10) -> list[dict]:
        """Read this session's experiments from MLflow, best validation AUC first."""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id],
                                  order_by=["metrics.val_auc DESC"])
        if runs.empty or "metrics.val_auc" not in runs.columns:
            return []
        runs = runs[runs["metrics.val_auc"].notna()]   # skip the FINAL-test run
        runs = runs[runs["run_id"].isin(self.history)]  # only this session's runs
        rows = []
        for _, r in runs.head(top_n).iterrows():
            rows.append({
                "rank": len(rows) + 1,
                "run_id": r["run_id"],
                "experiment_number": int(r.get("tags.experiment_number", 0) or 0),
                "model_type": r.get("tags.model_type"),
                "val_auc": r.get("metrics.val_auc"),
                "val_pr_auc": r.get("metrics.val_pr_auc"),
                "val_recall": r.get("metrics.val_recall"),
                "train_auc": r.get("metrics.train_auc"),
                "n_features": int(r.get("tags.n_features", 0) or 0),
                "params": {c[len("params."):]: r[c] for c in runs.columns
                           if c.startswith("params.") and pd.notna(r[c])},
                "notes": r.get("tags.notes", ""),
            })
        return rows

    # ======================= TOOL 4 =======================
    def final_test(self, run_id: str) -> dict:
        """Retrain the chosen config on train+validation, score on test. ONE TIME ONLY."""
        if self._test_used:
            return {"error": "Test set already used. It can only be used once."}
        if run_id not in self.history:
            return {"error": "Unknown run_id. Pick one from get_leaderboard()."}
        self._test_used = True

        cfg = self.history[run_id]
        full = pd.concat([self.train, self.val], ignore_index=True)
        X, y = full[cfg["features"]], full[self.target]
        pipe = Pipeline([("prep", _preprocessor(X)),
                         ("model", _build_model(cfg["model_type"], cfg["params"]))])
        pipe.fit(X, y)

        test = pd.read_csv(self._test_path)          # first and only read
        test_m = _metrics(test[self.target], pipe.predict_proba(test[cfg["features"]])[:, 1])

        model_path = ARTIFACTS / self.name / "final_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)

        with mlflow.start_run(experiment_id=self.experiment_id, run_name="FINAL-test"):
            mlflow.set_tags({"model_type": cfg["model_type"], "final": "true",
                             "source_run_id": run_id})
            mlflow.log_params({k: str(v) for k, v in cfg["params"].items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            mlflow.log_artifact(str(model_path))

        val_auc = mlflow.get_run(run_id).data.metrics["val_auc"]
        return {
            "chosen_run_id": run_id,
            "model_type": cfg["model_type"],
            "params": cfg["params"],
            "n_features": len(cfg["features"]),
            "val_auc": val_auc,
            "test_metrics": test_m,
            "val_to_test_auc_drop": round(val_auc - test_m["auc"], 4),
            "experiments_used": self.used,
            "model_saved_to": str(model_path),
        }


def as_json(obj) -> str:
    """Helper so results print nicely (and can be passed to the LLM as text)."""
    return json.dumps(obj, indent=2, default=lambda o: o.item()
                      if isinstance(o, np.generic) else str(o))
