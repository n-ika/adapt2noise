import pandas as pd

df = pd.read_pickle("timit.pkl")
df = df[df['phone'].isin(["eh","ae"])]

df.to_csv("voweldur.csv")