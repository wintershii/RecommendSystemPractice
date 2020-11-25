import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

csv_data = pd.read_csv(
    "C://Users//mirac//Desktop//课程资料//chapter_4//fillna.csv")
df = pd.DataFrame(csv_data)
# print(df)

# 固定值填充
# df['rating'] = df['rating'].fillna(0.0)
# print(df)

# 平均值填充
# df['rating'] = df['rating'].fillna(round(df['rating'].mean(), 1))
# print(df)

# 中位数填充
# df['rating'].fillna(round(df['rating'].median(), 1), inplace=True)
# print(df)

# 众数填充
# mode = df['rating'].dropna().mode().values
# df['rating'].fillna(mode[0], inplace=True)
# print(df)

# 使用SimpleImputer
X = df.iloc[:, 3:4].values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
df['rating'] = np.around(imp.transform(X), decimals=1)
print(df)
