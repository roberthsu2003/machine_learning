import pandas as pd
from sklearn.preprocessing import StandardScaler

# 建立一個範例 DataFrame
data = {'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 使用 StandardScaler ---
# 建立 StandardScaler 物件
scaler = StandardScaler()

# 對數據進行擬合與轉換
df_scaled = scaler.fit_transform(df)

# 將轉換後的數據轉換回 DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

print("標準化 (Standardization) 後的數據:")
print(df_scaled)
print("\n")

# 檢查轉換後數據的統計特性
print("標準化後數據的平均值:")
print(df_scaled.mean())
print("\n")

print("標準化後數據的標準差:")
print(df_scaled.std())
print("\n")

# --- 標準化的說明 ---
# 標準化是將數據轉換為平均值為 0，標準差為 1 的分佈。
# 公式為：z = (x - μ) / σ
# 其中 μ 是平均值，σ 是標準差。

# 適用時機：
# - 當特徵的數值範圍差異很大時。
# - 適用於許多機器學習演算法，例如：
#   - 線性迴歸 (Linear Regression)
#   - 邏輯迴歸 (Logistic Regression)
#   - 支援向量機 (Support Vector Machines, SVM)
#   - 主成分分析 (Principal Component Analysis, PCA)
# - 當數據的分佈接近常態分佈時，效果很好。
