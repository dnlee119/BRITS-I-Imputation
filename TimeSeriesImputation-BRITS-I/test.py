import pandas as pd
import numpy as np

np_1= np.array([1, 2, 3, 4, 5])
print(np_1)
dict = {"value":np_1.tolist()}
df_1 = pd.DataFrame(dict)
print(df_1)

path = "./data.csv"
df = pd.read_csv(path)
df_f = pd.concat([df["time"], df_1], axis=1, join='inner')
print(df_f)

ak = df["value"].isnull()
n = 0
for i in range(len(ak)):
    if ak.iloc[i] == True:
        df.iloc[i] = 1

print(df.iloc[100])