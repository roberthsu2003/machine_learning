import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# 建立一個包含缺失值的範例 DataFrame
data = {'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': ['apple', 'banana', 'apple', np.nan, 'banana']}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 方法一：刪除缺失值 ---
# dropna() 會刪除任何包含缺失值的行
df_dropped = df.dropna()
print("方法一：刪除缺失值後的數據:")
print(df_dropped)
print("\n")

# --- 方法二：填補缺失值 ---
# 使用 SimpleImputer 來填補缺失值
# 對於數值型數據，我們可以使用平均值 (mean)
imputer_mean = SimpleImputer(strategy='mean')
df_mean_imputed = df.copy()
df_mean_imputed[['A', 'B']] = imputer_mean.fit_transform(df_mean_imputed[['A', 'B']])
print("方法二：使用平均值填補數值型缺失值:")
print(df_mean_imputed)
print("\n")

# 對於類別型數據，我們可以使用最頻繁的值 (most_frequent)
imputer_most_frequent = SimpleImputer(strategy='most_frequent')
df_most_frequent_imputed = df.copy()
# SimpleImputer 需要 2D 陣列，所以我們需要 reshape
df_most_frequent_imputed['C'] = imputer_most_frequent.fit_transform(df_most_frequent_imputed['C'].values.reshape(-1, 1))
print("方法二：使用最頻繁值填補類別型缺失值:")
print(df_most_frequent_imputed)
print("\n")

# --- 結合填補 ---
df_imputed = df.copy()
df_imputed[['A', 'B']] = imputer_mean.fit_transform(df_imputed[['A', 'B']])
df_imputed['C'] = imputer_most_frequent.fit_transform(df_imputed['C'].values.reshape(-1, 1))
print("結合兩種填補方法:")
print(df_imputed)
