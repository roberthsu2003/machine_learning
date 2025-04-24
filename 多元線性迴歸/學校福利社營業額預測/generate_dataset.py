import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 設定隨機種子以確保可重現
np.random.seed(42)

# 生成日期（星期一至星期五，30週）
start_date = datetime(2025, 2, 3)  # 假設學期從2025年2月3日開始
dates = []
weekdays = []
for i in range(150):  # 150天，約30週
    current_date = start_date + timedelta(days=i)
    if current_date.weekday() < 5:  # 僅星期一至星期五
        dates.append(current_date.strftime("%Y-%m-%d"))
        weekdays.append(current_date.weekday() + 1)  # 1=星期一, 5=星期五

# 生成特徵數據
n_samples = len(dates)
data = {
    "日期": dates,
    "星期": weekdays,
    "氣溫": np.random.uniform(15, 35, n_samples),  # 氣溫15-35°C
    "降雨量": np.random.uniform(0, 50, n_samples),  # 降雨量0-50mm
    "請假人數": np.random.randint(0, 200, n_samples),  # 請假人數0-200
    "活動日": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # 20%機率有活動
}

# 生成營業額（模擬真實情況）
base_sales = 10000  # 基礎營業額
data["營業額"] = (
    base_sales
    + data["氣溫"] * 200  # 氣溫高，營業額略增
    - data["降雨量"] * 50  # 降雨量高，營業額下降
    - data["請假人數"] * 30  # 請假人數多，營業額下降
    + data["活動日"] * 3000  # 活動日營業額增加
    + np.random.normal(0, 1000, n_samples)  # 隨機噪聲
)
data["營業額"] = np.clip(data["營業額"], 5000, 20000)  # 限制範圍5000-20000

# 創建DataFrame
df = pd.DataFrame(data)

# 保存為CSV檔案
df.to_csv("Welfare_Club_Sales.csv", index=False, encoding="utf-8-sig")

print("資料集已生成並保存為 'Welfare_Club_Sales.csv'")