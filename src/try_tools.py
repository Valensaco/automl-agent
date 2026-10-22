"""Phase 2 check: run each tool by hand (no agents yet).

Trains one default version of each of the 4 models, then prints the
leaderboard read back from MLflow. Does NOT touch the test set.
"""
from tools import ExperimentSession, as_json

session = ExperimentSession.from_prepared_splits(budget=15, name="phase2-tool-check")

print("=== profile_data ===")
print(as_json(session.profile_data()))

print("\n=== train_and_evaluate (one default run per model) ===")
for model in ["logistic_regression", "random_forest", "lightgbm", "mlp"]:
    r = session.train_and_evaluate(model, notes="default settings")
    if "error" in r:
        print(f"{model:<20} ERROR: {r['error']}")
    else:
        print(f"{model:<20} val AUC {r['val_metrics']['auc']:.4f} | "
              f"train AUC {r['train_auc']:.4f} | {r['fit_seconds']}s | "
              f"budget left {r['budget_remaining']}")

print("\n=== error handling (bad setting, should NOT use budget) ===")
print(session.train_and_evaluate("lightgbm", params={"banana": 3}))

print("\n=== get_leaderboard (read back from MLflow) ===")
for row in session.get_leaderboard():
    print(f"#{row['rank']} {row['model_type']:<20} val AUC {row['val_auc']:.4f}")

print("\nAll tools work. Test set untouched.")
