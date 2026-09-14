import pandas as pd
df = pd.read_csv("data/raw/beauty_data.csv")
print(df.shape)
print(df.columns.tolist())
print(df.iloc[0])