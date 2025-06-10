import pandas as pd

df = pd.read_csv("superset.csv")

print(df['source'].unique())
print(df['post_author_country_location'].unique())
print(len(df['post_author_country_location'].unique().tolist()))