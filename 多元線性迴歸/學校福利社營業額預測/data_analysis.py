import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料集
df = pd.read_csv("福利社營業額資料集.csv")

# 設置繁體中文顯示
plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

# 1. 散點圖：氣溫 vs 營業額
plt.figure(figsize=(8, 6))
sns.scatterplot(x="氣溫", y="營業額", data=df)
plt.title("氣溫對營業額的影響")
plt.xlabel("氣溫 (°C)")
plt.ylabel("營業額 (新台幣)")
plt.savefig("散點圖_氣溫_營業額.png")
plt.close()

# 2. 箱形圖：星期 vs 營業額
plt.figure(figsize=(8, 6))
sns.boxplot(x="星期", y="營業額", data=df)
plt.title("不同星期對營業額的影響")
plt.xlabel("星期")
plt.ylabel("營業額 (新台幣)")
plt.savefig("箱形圖_星期_營業額.png")
plt.close()

# 3. 相關性熱圖
plt.figure(figsize=(8, 6))
corr = df[["氣溫", "降雨量", "請假人數", "活動日", "營業額"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("特徵相關性熱圖")
plt.savefig("相關性熱圖.png")
plt.close()

# 4. 直方圖：營業額分佈
plt.figure(figsize=(8, 6))
plt.hist(df["營業額"], bins=20, edgecolor="black")
plt.title("營業額分佈")
plt.xlabel("營業額 (新台幣)")
plt.ylabel("頻率")
plt.savefig("直方圖_營業額.png")
plt.close()

print("圖表已生成並保存")