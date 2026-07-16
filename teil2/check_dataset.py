import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "dataset_teil2.csv"

df = pd.read_csv(DATASET_PATH, sep=';')
print(df.head())

print()

print(df.info())
