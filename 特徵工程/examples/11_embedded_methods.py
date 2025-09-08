import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

# --- 嵌入法 (Embedded Methods) ---
# 嵌入法是將特徵選擇的過程，嵌入到模型的訓練過程中。
# 也就是說，模型在訓練的同時，也進行了特徵選擇。
# 這是過濾法和包裝法的一種折衷，通常有不錯的效果和效率。

# 建立一個分類問題的範例數據集
X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                           n_redundant=5, n_classes=2, random_state=42)

# 將數據轉換為 DataFrame (方便觀察)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

print("原始數據的維度:")
print(X.shape)
print("\n")

# --- 方法一：使用 Lasso (L1 正規化) ---
# Lasso (Least Absolute Shrinkage and Selection Operator) 是一種線性模型，
# 它在損失函數中加入 L1 正規化項，可以將不重要的特徵的係數縮減為 0。
# LassoCV 會自動幫我們找到最佳的 alpha (正規化強度) 參數。
lasso = LassoCV(cv=5, random_state=42).fit(X, y)

# 建立 SelectFromModel 物件
# threshold="median" 表示我們會選擇重要性高於中位數的特徵
# 這裡的重要性就是指 Lasso 模型的係數的絕對值
sfm_lasso = SelectFromModel(lasso, prefit=True, threshold="median")

# 轉換數據
X_selected_lasso = sfm_lasso.transform(X)

print("使用 Lasso 選擇特徵後的維度:")
print(X_selected_lasso.shape)
print("\n")

# 查看被選擇的特徵
selected_features_mask_lasso = sfm_lasso.get_support()
selected_features_lasso = X.columns[selected_features_mask_lasso]
print("Lasso 選擇的特徵:")
print(selected_features_lasso)
print("\n")


# --- 方法二：使用基於樹的模型 (例如隨機森林) ---
# 許多基於樹的模型 (例如隨機森林、梯度提升樹) 
# 在訓練後，可以提供特徵重要性 (feature importance)。
# 我們可以利用這個特性來進行特徵選擇。
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# 建立 SelectFromModel 物件
sfm_rf = SelectFromModel(rf, prefit=True, threshold="median")

# 轉換數據
X_selected_rf = sfm_rf.transform(X)

print("使用隨機森林選擇特徵後的維度:")
print(X_selected_rf.shape)
print("\n")

# 查看被選擇的特徵
selected_features_mask_rf = sfm_rf.get_support()
selected_features_rf = X.columns[selected_features_mask_rf]
print("隨機森林選擇的特徵:")
print(selected_features_rf)
