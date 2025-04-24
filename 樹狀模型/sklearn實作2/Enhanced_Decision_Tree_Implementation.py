import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 讀取資料集
data = pd.read_csv('Bookstore_Stationery_Dataset_Enhanced.csv')

# 資料預處理
le = LabelEncoder()
data['尺寸'] = le.fit_transform(data['尺寸'])
data['顏色'] = le.fit_transform(data['顏色'])
data['是否為書寫工具'] = le.fit_transform(data['是否為書寫工具'])
data['材質'] = le.fit_transform(data['材質'])
data['品牌'] = le.fit_transform(data['品牌'])

# 選擇特徵和目標變量
X = data[['價格', '尺寸', '顏色', '是否為書寫工具', '材質', '品牌']]
y = data['類型']

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定義決策樹參數網格
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用網格搜尋優化參數
clf = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)

# 最佳參數與模型
print("最佳參數:", clf.best_params_)
best_clf = clf.best_estimator_

# 預測
y_pred = best_clf.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"模型準確率: {accuracy:.2f}")
print("\n分類報告:")
print(classification_report(y_test, y_pred))

# 混淆矩陣圖表
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('混淆矩陣')
plt.xlabel('預測類型')
plt.ylabel('實際類型')
plt.savefig('confusion_matrix.png')

# 資料集圖表分析
plt.figure(figsize=(12, 6))

# 價格分佈
plt.subplot(1, 2, 1)
sns.histplot(data['價格'], kde=True)
plt.title('價格分佈圖')
plt.xlabel('價格')
plt.ylabel('數量')

# 類型分佈
plt.subplot(1, 2, 2)
sns.countplot(x='類型', data=data)
plt.title('類型分佈圖')
plt.xlabel('類型')
plt.ylabel('數量')

plt.tight_layout()
plt.savefig('data_analysis.png')