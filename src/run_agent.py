"""Run the AutoML agent from the terminal.

    python run_agent.py              # 15 experiments (default)
    python run_agent.py --budget 3   # quick, cheap test run
"""
import argparse
import asyncio

from agents import run_automl
from tools import ExperimentSession


def show(kind: str, d: dict):
    if kind == "start":
        print(f"Planner: {d['planner_model']} | Experimenter: {d['experimenter_model']} "
              f"| Budget: {d['budget']} experiments\n")
    elif kind == "experiment":
        r, p = d["result"], d["plan"]
        best = d["leaderboard"][0]
        print(f"[{r['experiment_number']:>2}] {r['model_type']:<20} "
              f"val AUC {r['val_metrics']['auc']:.4f} | gap {r['overfit_gap']:+.3f} "
              f"| best so far {best['val_auc']:.4f} | cost ${d['cost_usd']:.3f}")
        print(f"     why: {p.get('reasoning', '')}")
    elif kind == "error":
        print(f"  ! skipped: {d['error']}")
    elif kind == "final_choice":
        print(f"\nPlanner picks experiment #{d.get('experiment_number')}: {d.get('reason')}")
    elif kind == "done":
        f = d["final"]
        print("\n=== FINAL (test set used once) ===")
        print(f"Model: {f['model_type']} | val AUC {f['val_auc']:.4f} "
              f"| TEST AUC {f['test_metrics']['auc']:.4f}")
        print(f"Test metrics: {f['test_metrics']}")
        print(f"API cost: ${d['cost_usd']:.3f} | MLflow experiment: {d['experiment_name']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=15)
    args = ap.parse_args()
    session = ExperimentSession.from_prepared_splits(budget=args.budget)
    asyncio.run(run_automl(session, on_event=show))
