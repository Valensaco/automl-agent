"""Phase 1: download the UCI credit card default dataset and split it.

Split: 60% train / 20% validation / 20% test (stratified, fixed seed).
The test set is saved in data/locked/ and must only be used once, at the end.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
LOCKED = ROOT / "data" / "locked"

READABLE_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def download() -> pd.DataFrame:
    print("Downloading 'Default of Credit Card Clients' from UCI...")
    ds = fetch_ucirepo(id=350)
    X, y = ds.data.features.copy(), ds.data.targets.copy()
    if list(X.columns) == [f"X{i}" for i in range(1, 24)]:
        X.columns = READABLE_NAMES
    df = X.copy()
    df["default"] = y.iloc[:, 0].astype(int).values
    return df


def split(df: pd.DataFrame):
    train_val, test = train_test_split(
        df, test_size=0.20, stratify=df["default"], random_state=SEED)
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val["default"], random_state=SEED)
    return train, val, test


def main():
    for d in (RAW, SPLITS, LOCKED):
        d.mkdir(parents=True, exist_ok=True)

    df = download()
    df.to_csv(RAW / "credit_default.csv", index=False)

    train, val, test = split(df)
    train.to_csv(SPLITS / "train.csv", index=False)
    val.to_csv(SPLITS / "val.csv", index=False)
    test.to_csv(LOCKED / "test.csv", index=False)

    print(f"\nTotal rows: {len(df):,} | features: {df.shape[1] - 1}")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:<5} {len(part):>6,} rows | default rate {part['default'].mean():.1%}")
    print("\nDone. Test set is in data/locked/ (do not touch until the end).")


if __name__ == "__main__":
    main()
