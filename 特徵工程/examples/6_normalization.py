import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 建立一個範例 DataFrame
data = {'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 使用 MinMaxScaler ---
# 建立 MinMaxScaler 物件
# feature_range=(0, 1) 是預設值，表示將數據縮放到 0 到 1 之間
scaler = MinMaxScaler()

# 對數據進行擬合與轉換
df_scaled = scaler.fit_transform(df)

# 將轉換後的數據轉換回 DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("歸一化 (Normalization) 後的數據:")
print(df_scaled)
print("\n")

# 檢查轉換後數據的統計特性
print("歸一化後數據的最小值:")
print(df_scaled.min())
print("\n")

print("歸一化後數據的最大值:")
print(df_scaled.max())
print("\n")

# --- 歸一化的說明 ---
# 歸一化是將數據縮放到一個指定的範圍，通常是 [0, 1]。
# 公式為：x_scaled = (x - x_min) / (x_max - x_min)
# 其中 x_min 是最小值，x_max 是最大值。

# 適用時機：
# - 當你希望將數據限制在一個特定的範圍時。
# - 適用於一些對數據範圍敏感的演算法，例如：
#   - 類神經網路 (Neural Networks)
#   - K-近鄰演算法 (K-Nearest Neighbors, KNN)
# - 當數據的分佈不符合常態分佈，或者包含許多離群值時，
#   歸一化可能比標準化更穩定。
