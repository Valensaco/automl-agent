"""Copy the MLflow log into mlflow-space/ for the public Hugging Face site.

    python src/export_mlflow_snapshot.py                 # official run only
    python src/export_mlflow_snapshot.py --all           # every session

Paths inside the database are rewritten to the Space's folder, so the
artifacts (features list, final model, report) open correctly online.
"""
import argparse
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DB, SRC_ART = ROOT / "mlflow.db", ROOT / "mlartifacts"
OUT = ROOT / "mlflow-space"
SPACE_DIR = "/tmp/snap"      # where the Space's Dockerfile copies the snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="keep every agent session")
    args = ap.parse_args()

    db = OUT / "mlflow.db"
    shutil.copy(SRC_DB, db)
    con = sqlite3.connect(db)
    cur = con.cursor()

    sessions = cur.execute("""
        SELECT e.experiment_id, e.name, COUNT(r.run_uuid) FROM experiments e
        LEFT JOIN runs r ON r.experiment_id = e.experiment_id
        WHERE e.name LIKE 'session-%' AND e.lifecycle_stage = 'active'
        GROUP BY e.experiment_id ORDER BY COUNT(r.run_uuid) DESC, e.name DESC""").fetchall()
    if not sessions:
        raise SystemExit("No agent sessions found in mlflow.db.")
    keep = [s for s in sessions] if args.all else sessions[:1]
    keep_ids = {str(s[0]) for s in keep}

    # hide everything else (tests, tool checks, the Default experiment)
    cur.execute("UPDATE experiments SET lifecycle_stage='deleted' WHERE CAST(experiment_id AS TEXT) "
                f"NOT IN ({','.join('?' * len(keep_ids))})", tuple(keep_ids))

    # rewrite artifact paths: file:///Users/.../mlartifacts/x -> file:///tmp/snap/mlartifacts/x
    old, new = SRC_ART.as_uri(), Path(SPACE_DIR, "mlartifacts").as_uri()
    cur.execute("UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, ?)", (old, new))
    cur.execute("UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, ?)", (old, new))
    con.commit()
    con.execute("VACUUM")
    con.close()

    shutil.rmtree(OUT / "mlartifacts", ignore_errors=True)
    for _, name, _ in keep:
        if (SRC_ART / name).exists():
            shutil.copytree(SRC_ART / name, OUT / "mlartifacts" / name)

    for _, name, n in keep:
        print(f"Included: {name} ({n} runs)")
    print(f"Snapshot written to {OUT}")


if __name__ == "__main__":
    main()
