import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 建立一個包含類別數據的範例 DataFrame
data = {'color': ['Red', 'Green', 'Blue', 'Green', 'Red']}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 方法一：使用 scikit-learn 的 OneHotEncoder ---
# 建立 OneHotEncoder 物件
# sparse=False 表示我們想要得到一個 NumPy 陣列，而不是稀疏矩陣
ohe = OneHotEncoder(sparse_output=False)

# 對 'color' 欄位進行擬合與轉換
# fit_transform 需要 2D 陣列，所以我們使用 [['color']]
encoded_features = ohe.fit_transform(df[['color']])

# 將轉換後的特徵轉換為 DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(['color']))

# 將編碼後的特徵與原始數據合併
df_ohe = pd.concat([df, encoded_df], axis=1)

print("使用 scikit-learn 的 OneHotEncoder 轉換後的數據:")
print(df_ohe)
print("\n")


# --- 方法二：使用 pandas 的 get_dummies ---
# get_dummies 是一個更簡單直接的方法
df_dummies = pd.get_dummies(df, columns=['color'], prefix='color')

print("使用 pandas 的 get_dummies 轉換後的數據:")
print(df_dummies)
print("\n")

# --- One-Hot Encoding 的優缺點 ---
# 優點：
# - 解決了 Label Encoding 引入不存在順序關係的問題。
# - 每個類別都是獨立的特徵，模型不會對它們進行排序。

# 缺點：
# - 當類別數量很多時，會產生大量的特徵 (維度災難)。
# - 可能會導致數據稀疏，增加計算成本。
