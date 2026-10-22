"""Phase 5: turn a finished agent run into a readable report.

Writes reports/<session>.md (for people) and reports/<session>.json (for the
Streamlit app's demo mode), and attaches both to the session in MLflow.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import mlflow

from tools import ROOT, TRACKING_URI, as_json

REPORTS = ROOT / "reports"

PRETTY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
          "lightgbm": "LightGBM", "mlp": "Neural network (MLP)"}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def build_markdown(s: dict) -> str:
    f, p = s["final"], s["profile"]
    cm = f["confusion_matrix"]
    t, d = f["test_metrics"], f["test_metrics_default_0_5"]
    defaulters = cm["true_positive"] + cm["false_negative"]
    flagged = cm["true_positive"] + cm["false_positive"]

    lines = [
        f"# AutoML Agent Report: {s['experiment_name']}",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "## Summary",
        f"- **Dataset:** {p['rows_train'] + p['rows_validation'] + f['test_rows']:,} rows, "
        f"{p['n_features']} features, target `{p['target']}` "
        f"(positive rate {_pct(p['positive_rate'])})",
        f"- **Split:** {p['rows_train']:,} train / {p['rows_validation']:,} validation / "
        f"{f['test_rows']:,} test (test used once, at the end)",
        f"- **Budget:** {f['experiments_used']} experiments | **API cost:** ${s['cost_usd']:.3f}",
        f"- **Final model:** {PRETTY.get(f['model_type'], f['model_type'])} "
        f"(experiment #{s['choice'].get('experiment_number')})",
        f"- **Validation AUC:** {f['val_auc']:.4f} | **Test AUC:** {t['auc']:.4f} "
        f"(drop {f['val_to_test_auc_drop']:+.4f})",
        "",
        "## Why the planner chose this model",
        f"> {s['choice'].get('reason', '')}",
        "",
        "## Test results",
        f"The decision cutoff was tuned on **validation** data to maximize F1 "
        f"(cutoff = **{f['threshold']:.2f}** instead of the default 0.50).",
        "",
        "| Metric | Default cutoff 0.50 | Tuned cutoff " + f"{f['threshold']:.2f} |",
        "|---|---|---|",
        *[f"| {name} | {d[k]:.4f} | {t[k]:.4f} |" for k, name in
          [("auc", "AUC-ROC"), ("pr_auc", "PR-AUC"), ("recall", "Recall (defaulters caught)"),
           ("precision", "Precision (flags that were right)"), ("f1", "F1")]],
        "",
        "**In plain words (test set, tuned cutoff):**",
        f"- Of {defaulters:,} customers who actually defaulted, the model caught "
        f"**{cm['true_positive']:,}** ({_pct(t['recall'])}) and missed {cm['false_negative']:,}.",
        f"- It flagged {flagged:,} customers as risky; **{cm['true_positive']:,}** of them "
        f"really defaulted ({_pct(t['precision'])}).",
        "",
        "| | Predicted: no default | Predicted: default |",
        "|---|---|---|",
        f"| **Actual: no default** | {cm['true_negative']:,} | {cm['false_positive']:,} |",
        f"| **Actual: default** | {cm['false_negative']:,} | {cm['true_positive']:,} |",
        "",
        "## Most important features",
        "Permutation importance on validation: how much AUC drops when a feature is shuffled.",
        "",
        "| Feature | AUC drop |",
        "|---|---|",
        *[f"| `{r['feature']}` | {r['auc_drop']:.4f} |" for r in f["feature_importance"]],
        "",
        "## The agent's experiments, step by step",
        "",
        "| # | Model | Val AUC | Overfit gap | Planner's reasoning |",
        "|---|---|---|---|---|",
    ]
    for step in s["steps"]:
        r = step["result"]
        why = step["plan"].get("reasoning", "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {r['experiment_number']} | {PRETTY.get(r['model_type'], r['model_type'])} "
                     f"| {r['val_metrics']['auc']:.4f} | {r['overfit_gap']:+.3f} | {why} |")

    lines += [
        "",
        "## Final settings",
        "```json",
        json.dumps(f["params"], indent=2, default=str),
        "```",
        "",
        "## How the agent works",
        "- **Planner** (Claude Sonnet, via AutoGen) reads the data profile and the MLflow "
        "leaderboard and chooses the next experiment. It has no tools.",
        "- **Experimenter** (Claude Haiku, via AutoGen) runs the plan by calling "
        "`train_and_evaluate`, which logs every run to MLflow.",
        "- **Check:** the test set is only loaded by `final_test()`, which the code (not the "
        "LLM) calls once after the budget is used up.",
    ]
    return "\n".join(lines)


def write_report(summary: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    name = summary["experiment_name"]
    md_path, json_path = REPORTS / f"{name}.md", REPORTS / f"{name}.json"
    md_path.write_text(build_markdown(summary))
    json_path.write_text(as_json(summary))

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(name)
    if exp:
        with mlflow.start_run(experiment_id=exp.experiment_id, run_name="REPORT"):
            mlflow.log_artifact(str(md_path))
            mlflow.log_artifact(str(json_path))
    return md_path
