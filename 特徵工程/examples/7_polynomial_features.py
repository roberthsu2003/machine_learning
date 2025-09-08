import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# 建立一個範例 DataFrame
data = {'x1': [1, 2, 3],
        'x2': [4, 5, 6]}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 使用 PolynomialFeatures ---
# 建立 PolynomialFeatures 物件
# degree=2 表示我們想要建立二次多項式特徵
# include_bias=False 表示我們不想要包含偏差項 (常數 1)
poly = PolynomialFeatures(degree=2, include_bias=False)

# 對數據進行擬合與轉換
poly_features = poly.fit_transform(df)

# 將轉換後的特徵轉換為 DataFrame
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df.columns))

print("建立二次多項式特徵後的數據:")
print(poly_df)
print("\n")

# --- 多項式特徵的說明 ---
# 多項式特徵可以幫助模型捕捉特徵之間的非線性關係。
# 在這個範例中，我們有兩個原始特徵 x1 和 x2。
# degree=2 的多項式特徵包括：
# - x1, x2 (原始特徵)
# - x1^2, x2^2 (二次項)
# - x1*x2 (互動項)

# 適用時機：
# - 當你懷疑特徵之間存在非線性關係時。
# - 常用於線性模型 (例如線性迴歸)，以增加模型的複雜度，
#   使其能夠擬合更複雜的數據。

# 注意事項：
# - degree 的選擇很重要，過高的 degree 可能會導致模型過擬合。
# - 多項式特徵會顯著增加特徵的數量，可能會增加計算成本。
