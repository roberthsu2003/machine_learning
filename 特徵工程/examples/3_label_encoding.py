import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 建立一個包含類別數據的範例 DataFrame
data = {'size': ['S', 'M', 'L', 'XL', 'S', 'M']}
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 使用 LabelEncoder ---
# 建立 LabelEncoder 物件
le = LabelEncoder()

# 對 'size' 欄位進行擬合與轉換
df['size_encoded'] = le.fit_transform(df['size'])

print("使用 LabelEncoder 轉換後的數據:")
print(df)
print("\n")

# 查看編碼的對應關係
# classes_ 屬性會顯示原始的類別
print("編碼的對應關係:")
for i, label in enumerate(le.classes_):
    print(f"{label} -> {i}")

# --- Label Encoding 的優缺點 ---
# 優點：
# - 實現簡單，計算速度快。
# - 節省空間，因為只用一個欄位來表示。

# 缺點：
# - 引入了不存在的順序關係。
#   例如，在這個範例中，L=0, M=1, S=2, XL=3。
#   模型可能會誤認為 S > M > L，這在很多情況下是不正確的。
# - 因此，Label Encoding 通常適用於有序的類別數據 (Ordinal Data)，
#   例如 "低"、"中"、"高"。
