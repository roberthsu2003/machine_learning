import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 載入資料集
df = pd.read_csv("福利社營業額資料集.csv")

# 選擇特徵和目標變量
X = df[["星期", "氣溫", "降雨量", "請假人數", "活動日"]]
y = df["營業額"]

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. 無正規化版本
model_no_norm = LinearRegression()
model_no_norm.fit(X_train, y_train)
y_pred_no_norm = model_no_norm.predict(X_test)

# 計算評估指標
mse_no_norm = mean_squared_error(y_test, y_pred_no_norm)
rmse_no_norm = np.sqrt(mse_no_norm)
r2_no_norm = r2_score(y_test, y_pred_no_norm)

print("無正規化模型結果：")
print(f"均方誤差 (MSE): {mse_no_norm:.2f}")
print(f"均方根誤差 (RMSE): {rmse_no_norm:.2f}")
print(f"R² 分數: {r2_no_norm:.2f}")
print("模型係數：", model_no_norm.coef_)
print("截距：", model_no_norm.intercept_)
print()

# 2. 有正規化版本（標準化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_norm = LinearRegression()
model_norm.fit(X_train_scaled, y_train)
y_pred_norm = model_norm.predict(X_test_scaled)

# 計算評估指標
mse_norm = mean_squared_error(y_test, y_pred_norm)
rmse_norm = np.sqrt(mse_norm)
r2_norm = r2_score(y_test, y_pred_norm)

print("有正規化模型結果：")
print(f"均方誤差 (MSE): {mse_norm:.2f}")
print(f"均方根誤差 (RMSE): {rmse_norm:.2f}")
print(f"R² 分數: {r2_norm:.2f}")
print("模型係數：", model_norm.coef_)
print("截距：", model_norm.intercept_)