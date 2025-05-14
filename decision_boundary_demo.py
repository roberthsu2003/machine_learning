import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 生成資料集
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# 創建圖形
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 1. 繪製原始資料點
ax1.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='類別 0')
ax1.scatter(X[y==1, 0], X[y==1, 1], c='red', label='類別 1')
ax1.set_title('步驟1: 原始資料點')
ax1.set_xlabel('特徵 0')
ax1.set_ylabel('特徵 1')
ax1.legend()

# 2. 繪製網格點
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

ax2.scatter(xx, yy, c='gray', alpha=0.1, s=1)
ax2.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='類別 0')
ax2.scatter(X[y==1, 0], X[y==1, 1], c='red', label='類別 1')
ax2.set_title('步驟2: 生成網格點')
ax2.set_xlabel('特徵 0')
ax2.set_ylabel('特徵 1')
ax2.legend()

# 3. 繪製決策邊界
model = LogisticRegression()
model.fit(X, y)

# 對網格點進行預測
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製決策邊界和資料點
ax3.contour(xx, yy, Z, alpha=0.7)
ax3.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='類別 0')
ax3.scatter(X[y==1, 0], X[y==1, 1], c='red', label='類別 1')
ax3.set_title('步驟3: 決策邊界')
ax3.set_xlabel('特徵 0')
ax3.set_ylabel('特徵 1')
ax3.legend()

plt.tight_layout()
plt.show()

# 印出一些網格點的預測結果
print("網格點形狀:", xx.shape)
print("\n前5x5網格點的預測結果:")
print(Z[:5, :5]) 