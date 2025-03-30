# 訓練流程, 模型的泛化(Generalization),過渡擬合(Overfitting)和欠擬合(Underfitting)
### ML的整個「訓練過程」：這裡以監督式學習(Supervised Learning)為例

| **階段**   | **要做的事**  | **說明**                                  |
| ----------- | --------- | --------------------------------------- |
| 1. (訓練前)  | 決定資料集與分析資料      | 你想要預測的是什麼資料? 這邊需要先知道 example、label、features的概念。 |
| 2. (訓練前) | 決定問題種類   | 依據資料，會知道是什麼類型的問題。regression problem(回歸問題)? classification problem(分類問題)?          |
| 3. (訓練前)  | 決定ML模型(ML models)     | 依據問題的種類，會知道需要使用什麼對應的ML模型。回歸模型(Regression model)? 分類模型(Classification model)? |
| 4. (訓練前) | (模型裡面的參數)     | ML模型裡面的參數(parameters)與超參數(hyper-parameters)   |
| 5. (訓練中) 調整模型  | 評估當前模型好壞    | 損失函數(Loss Functions)：使用損失函數評估目前模型的好與壞。以MSE(Mean Squared Error), RMSE(Root Mean Squared Error), 交叉熵(Cross Entropy)為例。 |
| (訓練中) 調整模型  | 修正模型參數    | 以梯度下降法 (Gradient Descent)為例：決定模型中參數的修正「方向」與「步長(step size)」 |
| 7. (訓練中) 調整腳步 | 調整學習腳步 | 透過學習速率(learning rate)來調整ML模型訓練的步長(step size)，調整學習腳步。|
| 8. (訓練中) 加快訓練 | 取樣與分堆 | 設定batch size，透過batch從訓練目標中取樣，來加快ML模型訓練的速度。(此參數在訓練前設定，為hyper-parameter)。與迭代(iteration),epoch介紹。 |
| 9. ((訓練中) 完成訓練 | (loop) -> 完成 | 重覆過程(評估當前模型好壞 -> 修正模型參數)，直到能通過「驗證資料集(Validation)」的驗證即可結束訓練。 |
| 10. (訓練後) | 訓練結果可能問題 | 「不適當的最小loss?」 |
| 11. (訓練後) | 訓練結果可能問題 | 欠擬合(underfitting)?過度擬合(overfitting)?  |
| 12. (訓練後) | 評估 - 性能指標 | 性能指標(performance metrics)：以混淆矩陣(confusion matrix)分析，包含「Accuracy」、「Precision」、「Recall」三種評估指標。  |
| 13. (訓練後) | 評估 - 新資料適用性 | 泛化(Generalization)：對於新資料、沒看過的資料的模型適用性。 |
| 14. (訓練後) | 評估 - 模型測試 | 使用「獨立測試資料集(Test)」測試. 使用交叉驗證(cross-validation)(又稱bootstrapping)測試. |

### 泛化(generalize)：指ML模型「對未知資料集」的預測能力。

> 泛化(generalize)能力差：等於預測「對未知資料集」的預測能力能力差。  
> 但如果對「對自己的資料」的預測能力很好，有可能是發生了過度擬合(overfitting)的現象。

### ★ 欠擬合(underfitting) 與 過度擬合(overfitting)


| **比較**   | **‌ 欠擬合(underfitting)**  | ** 過度擬合(overfitting)** |
| ----------- | --------- | --------------------------------------- |
| (訓練前) | (可能)決定了太簡單的模型 | (可能)決定了太複雜的模型 |
| (訓練中) | (可能)訓練太早結束 | (可能)訓練過頭，也就是太晚結束 |
| (訓練後)對自已的資料 | 訓練後發現模型「對自已的資料」預測能力太差 | 訓練後發現模型「對自已的資料」預測能力非常好(可能好到沒有誤差) |
| (訓練後)對新的資料 | (對自己的資料都不行了還要試新資料嗎XD) | 訓練後發現模型「對新的資料」預測能力非常差 |
| 代表的意義 | 我們的模型「對自已的資料」沒辦法達到理想的預測能力 | 我們的模型「對新的資料」沒辦法達到理想的預測能力，然而對「對自已的資料」預測能力非常好。 |

> [!TIP]
> 「最佳的ML模型訓練結果」應該介於欠擬合(underfitting) 與 過度擬合(overfitting)之間。

#### 用圖簡單解釋 欠擬合(underfitting) 與 過度擬合(overfitting)

使用Matplotlib 來展示欠擬合 (underfitting) 和過度擬合 (overfitting) 的概念，並搭配簡單的例子來說明這兩個現象在機器學習中的區別。

假設我們有一個簡單的非線性數據集，並嘗試用不同複雜度的模型來擬合它：

[**實作的ipynb**](./README.ipynb)

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成模擬數據
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)  # 真實數據為正弦函數加上噪聲

# 欠擬合：用一次多項式（直線）擬合
underfit_coeffs = np.polyfit(x, y, 1)  # 一次多項式
underfit_model = np.poly1d(underfit_coeffs)

# 過度擬合：用高次多項式（15次）擬合
overfit_coeffs = np.polyfit(x, y, 15)  # 15次多項式
overfit_model = np.poly1d(overfit_coeffs)

# 適當擬合：用三次多項式擬合
goodfit_coeffs = np.polyfit(x, y, 3)  # 三次多項式
goodfit_model = np.poly1d(goodfit_coeffs)

# 繪圖
plt.figure(figsize=(12, 8))

# 原始數據
plt.scatter(x, y, color='gray', label='數據點', alpha=0.5)

# 欠擬合
plt.plot(x, underfit_model(x), color='blue', label='欠擬合 (1次多項式)', linewidth=2)

# 過度擬合
plt.plot(x, overfit_model(x), color='red', label='過度擬合 (15次多項式)', linewidth=2)

# 適當擬合
plt.plot(x, goodfit_model(x), color='green', label='適當擬合 (3次多項式)', linewidth=2)

# 圖表設置
plt.title('欠擬合 vs 過度擬合 vs 適當擬合')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
```

### 解釋：
1. **欠擬合 (Underfitting)**：
   - 用藍色直線表示（一次多項式）。
   - 模型過於簡單，無法捕捉數據中的非線性模式（這裡是正弦波）。
   - 結果是模型既不能很好地擬合訓練數據，也無法泛化到新數據。

2. **過度擬合 (Overfitting)**：
   - 用紅色曲線表示（15次多項式）。
   - 模型過於複雜，不僅擬合了數據的真實趨勢，還擬合了噪聲。
   - 雖然在訓練數據上表現很好，但在新數據上泛化能力差。

3. **適當擬合 (Good Fit)**：
   - 用綠色曲線表示（三次多項式）。
   - 模型複雜度適中，能夠捕捉數據的主要趨勢（正弦波），同時不過分擬合噪聲。
   - 在訓練數據和新數據上都能表現良好。

### 圖表展示：
運行這段程式碼後，你會看到一個圖表，灰色點是原始數據，藍線是欠擬合，紅線是過度擬合，綠線是適當擬合。這樣的視覺化可以清楚展示這三者的區別。

