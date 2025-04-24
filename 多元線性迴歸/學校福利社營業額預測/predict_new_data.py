import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 載入資料集
df = pd.read_csv("福利社營業額資料集.csv")

# 選擇特徵和目標變量
X = df[["星期", "氣溫", "降雨量", "請假人數", "活動日"]]
y = df["營業額"]

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練正規化模型
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 生成3筆新數據
new_data = pd.DataFrame({
    "星期": [1, 3, 5],  # 星期一、星期三、星期五
    "氣溫": [25.5, 30.0, 20.5],  # 合理氣溫範圍
    "降雨量": [0.0, 10.0, 30.0],  # 不同降雨量
    "請假人數": [50, 20, 100],  # 合理請假人數
    "活動日": [0, 1, 0]  # 無活動、有活動、無活動
})

# 對新數據進行標準化
new_data_scaled = scaler.transform(new_data)

# 預測營業額
predictions = model.predict(new_data_scaled)

# 將預測結果加入新數據
new_data["預測營業額"] = np.round(predictions, 2)

# 輸出結果
print("新數據及其預測營業額：")
print(new_data.to_string(index=False))

# 保存結果為CSV（可選）
new_data.to_csv("新數據預測結果.csv", index=False, encoding="utf-8-sig")