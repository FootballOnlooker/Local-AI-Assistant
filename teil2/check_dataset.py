from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset_teil2.csv"


def check_dataset() -> None:
    """Display basic dataset quality information."""

    df = pd.read_csv(DATASET_PATH, sep=";")

    print("First rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    check_dataset()