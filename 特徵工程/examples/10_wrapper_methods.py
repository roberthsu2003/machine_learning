import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# --- 包裝法 (Wrapper Methods) ---
# 包裝法是將特徵選擇的過程，看作是一個搜尋問題。
# 它會使用一個機器學習模型來評估不同特徵子集的好壞。
# 效果通常比過濾法好，但計算成本更高。

# 建立一個分類問題的範例數據集
X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                           n_redundant=5, n_classes=2, random_state=42)

# 將數據轉換為 DataFrame (方便觀察)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

print("原始數據的維度:")
print(X.shape)
print("\n")

# --- 使用 RFE (Recursive Feature Elimination) ---
# RFE (遞歸特徵消除) 是一種常用的包裝法。
# 它的原理是：
# 1. 使用所有特徵來訓練一個模型。
# 2. 根據模型給出的特徵重要性 (例如係數或權重)，
#    移除最不重要的特徵。
# 3. 重複以上過程，直到剩下指定數量的特徵。

# 建立一個模型 (estimator)
# 這裡我們使用邏輯迴歸模型
model = LogisticRegression(solver='liblinear')

# 建立 RFE 物件
# estimator: 用來評估特徵重要性的模型
# n_features_to_select: 要選擇的特徵數量
rfe = RFE(estimator=model, n_features_to_select=5)

# 對數據進行擬合
rfe.fit(X, y)

# 轉換數據
X_selected = rfe.transform(X)

print("使用 RFE 選擇 5 個特徵後的維度:")
print(X_selected.shape)
print("\n")

# --- 查看被選擇的特徵 ---
# support_ 屬性會返回一個布林遮罩
selected_features_mask = rfe.support_
selected_features = X.columns[selected_features_mask]

print("被選擇的特徵:")
print(selected_features)
print("\n")

# ranking_ 屬性會顯示每個特徵的排名
# 排名為 1 的是最好的特徵
feature_ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
feature_ranking = feature_ranking.sort_values(by='Ranking')
print("每個特徵的排名 (1 是最好):")
print(feature_ranking)
