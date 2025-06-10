import pandas as pd

df = pd.read_json("datasets/explain.json").T

print(df.head())
print(df.columns.tolist())
print(df.iloc[0]['annotators'])