import pandas as pd

df = pd.read_csv("data/dataset_teil2.csv", sep=';')
print(df.head())

print()

print(df.info())
