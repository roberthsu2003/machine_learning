import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# 建立一個範例 DataFrame
data = {'x1': [1, 2, 3],
        'x2': [4, 5, 6],
        'x3': [7, 8, 9]}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 使用 PolynomialFeatures 建立互動特徵 ---
# 建立 PolynomialFeatures 物件
# interaction_only=True 表示我們只想要互動特徵，不要二次項 (例如 x1^2)
# include_bias=False 表示我們不想要包含偏差項 (常數 1)
poly = PolynomialFeatures(interaction_only=True, include_bias=False)

# 對數據進行擬合與轉換
interaction_features = poly.fit_transform(df)

# 將轉換後的特徵轉換為 DataFrame
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out(df.columns))

print("只建立互動特徵後的數據:")
print(interaction_df)
print("\n")

# --- 互動特徵的說明 ---
# 互動特徵是兩個或多個特徵的乘積。
# 它可以幫助模型捕捉特徵之間的協同效應。
# 例如，房價可能不僅取決於房間數量和房屋大小，
# 還取決於這兩者的互動 (例如，大房屋的每個房間可能更有價值)。

# 在這個範例中，我們有三個原始特徵 x1, x2, x3。
# 互動特徵包括：
# - x1, x2, x3 (原始特徵)
# - x1*x2, x1*x3, x2*x3 (兩個特徵的互動)

# 適用時機：
# - 當你根據領域知識，認為某些特徵的組合具有特殊的意義時。
# - 常用於線性模型，以增加模型的表達能力。

# 與多項式特徵的比較：
# - PolynomialFeatures(degree=2) 會產生 x1, x2, x1^2, x1*x2, x2^2
# - PolynomialFeatures(interaction_only=True) 會產生 x1, x2, x1*x2
# - 互動特徵是多項式特徵的一個子集。
