# 📊 數據基礎與 NumPy 實戰：機器學習的起點

> [!IMPORTANT]
> **本章節學習目標**：
> 1. **了解 ML 數據結構**：搞懂什麼是「特徵矩陣 $X$」與「標籤向量 $y$」，以及關鍵維度 `shape`。
> 2. **掌握核心運算套件**：學會機器學習的基石 **NumPy** 最常用的 5 大操作（複製貼上即可執行）。
> 3. **經典範例數據實戰**：透過 Scikit-learn 與 mglearn 精選數據集，體會「分類 (Classification)」與「迴歸 (Regression)」任務的本質。

---

## 🧰 1. 認識機器學習的五大核心工具箱

在學習機器學習時，不需要從頭手寫底層演算法，社群已經為我們準備好了強大的函式庫：

| 套件名稱 | 直觀角色比喻 | 在機器學習中的主要任務 | 常用縮寫或載入方式 |
| :--- | :--- | :--- | :--- |
| **`NumPy`** | 🧮 **超級矩陣計算機** | 高效率的多維陣列運算、矩陣乘法，所有模型底層運算的基石。 | `import numpy as np` |
| **`Pandas`** | 📋 **Python 版 Excel** | 讀取 CSV/Excel、表格數據整理、缺失值處理與欄位篩選。 | `import pandas as pd` |
| **`Matplotlib`** | 🎨 **數據繪圖筆** | 將特徵分佈、擬合直線、決策邊界繪製成直觀的圖表。 | `import matplotlib.pyplot as plt` |
| **`Scikit-learn`** | 🤖 **機器學習百寶袋** | 提供數十種標準機器學習模型、評估指標與練習數據集。 | `import sklearn` |
| **`mglearn`** | 💡 **教學視覺化小助手** | 《Python 機器學習》教科書的教學專用套件，方便快速觀察數據邊界。 | `import mglearn` |

> [!TIP]
> **💡 Google Colab 中文顯示解法**：  
> 在 Colab 雲端環境中繪製 Matplotlib 圖表時，中文預設會變成方框（豆腐字）。只要在 Notebook 一開始執行中文字型下載與設定即可：
> ```python
> !pip install -q wget
> import os, wget, matplotlib.pyplot as plt, matplotlib.font_manager as fm
> 
> if not os.path.exists("ChineseFont.ttf"):
>     wget.download("https://github.com/roberthsu2003/machine_learning/raw/refs/heads/main/source_data/ChineseFont.ttf")
> fm.fontManager.addfont("ChineseFont.ttf")
> plt.rcParams['font.family'] = fm.FontProperties(fname="ChineseFont.ttf").get_name()
> ```

---

## 📐 2. 機器學習中的「數據」長怎樣？（核心觀念）

初學者最常遇到的困惑是：**「為什麼數據不是直接丟進去，而是要分 $X$ 和 $y$？」**

```
+-------------------------------------------------------------+
|                     特徵矩陣 X (Features)                   |  ==> 預測答案 y (Label)
|            [ 欄位 1: 身高,   欄位 2: 體重,   欄位 3: 年齡 ]   |      [ 類別: 性別 / 數值: 薪資 ]
+-------------------------------------------------------------+-------------------------+
| 第 1 筆:   [    175,             70,             25     ]   |  ==>       0 (男)
| 第 2 筆:   [    160,             50,             28     ]   |  ==>       1 (女)
| 第 3 筆:   [    180,             85,             35     ]   |  ==>       0 (男)
+-------------------------------------------------------------+-------------------------+
                     形狀 (Shape) = (3, 3)                              形狀 (Shape) = (3,)
```

1. **特徵矩陣 $X$ (Features)**：
   * **通常是二維陣列 (2D Array)**，形狀表示為 `(樣本數, 特徵數)`，即 `(幾筆資料, 幾個欄位)`。
   * 大多數機器學習演算法都強制要求 $X$ 必須是 2D 形狀！
2. **目標標籤 $y$ (Labels / Targets)**：
   * **通常是一維陣列 (1D Array)**，形狀表示為 `(樣本數,)`，代表每筆資料對應的標準答案。
3. **兩大任務類型**：
   * **分類 (Classification)**：$y$ 是離散的「類別」（如：0 或 1、良性或惡性、貓或狗）。
   * **迴歸 (Regression)**：$y$ 是連續的「數值」（如：房價、薪資、氣溫）。

---

## 💻 3. NumPy 初學者必備的 5 大操作（複製貼上專區）

> 🚀 **完整互動教學 Notebook**：  
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E6%95%B8%E6%93%9A%E5%9F%BA%E7%A4%8E%E8%88%87NumPy%E5%AF%A6%E6%88%B0/README.ipynb) [開啟 README.ipynb](./README.ipynb)

#### 🔹 技巧 1：建立二維陣列與檢查形狀（Shape）
```python
import numpy as np

# 建立 2 維陣列（模擬 3 筆資料，每筆 2 個特徵）
X = np.array([
    [1.5, 2.0],
    [3.2, 4.1],
    [5.0, 6.3]
])

print("資料內容:\n", X)
print("形狀 (樣本數, 特徵數):", X.shape)  # 輸出: (3, 2)
```

#### 🔹 技巧 2：切片取值（Slicing）- 擷取特定特徵欄位
```python
# 語法規則：陣列[列索引, 欄索引]，用冒號 : 代表「選取全部」
first_feature = X[:, 0]    # 選取所有樣本的「第 1 個特徵」
print("第 1 個特徵的所有數值:", first_feature)  # 輸出: [1.5, 3.2, 5.0]

subset = X[:2, :]          # 取出前 2 筆資料的所有特徵
print("前 2 筆資料:\n", subset)
```

#### 🔹 技巧 3：條件篩選（Boolean Masking）
```python
# 找出第 1 個特徵數值大於 2.0 的資料
mask = X[:, 0] > 2.0
filtered_X = X[mask]
print("符合條件 (第 1 特徵 > 2.0) 的資料:\n", filtered_X)
```

#### 🔹 技巧 4：廣播機制（Broadcasting）- 快速去中心化或標準化
```python
# 計算每個特徵欄位的平均值（axis=0 代表跨列直向計算）
col_means = X.mean(axis=0)
print("各欄位平均值:", col_means)

# NumPy 會自動將 (3, 2) 的 X 與 (2,) 的平均值對齊相減（廣播機制）
X_centered = X - col_means
print("去中心化後的數據:\n", X_centered)
```

#### 🔹 技巧 5：重塑形狀（Reshape）- 修正維度不合報錯
```python
# 當模型要求 2D 矩陣，而手邊資料只有 1D 時：
y_1d = np.array([10, 20, 30])
print("原始形狀:", y_1d.shape)  # (3,)

# 轉為二維直行矩陣 (3 筆資料, 1 個特徵)
y_2d = y_1d.reshape(-1, 1)
print("重塑後的形狀:", y_2d.shape)  # (3, 1)
```

---

## 🧪 4. 經典數據集實戰沙盒 (Dataset Sandboxes)

為了讓同學清楚「不同數據特性適合哪種任務」，我們將範例 Notebook 依照**分類**與**迴歸**兩大核心任務分類整理：

### 🎯 任務類型 A：分類問題 (Classification)
> **目標**：讓演算法學會將資料劃分至正確的群體或類別。

| 數據集名稱 | 資料規模與維度 | 教學特色與重點 | 線上執行與檔案 |
| :--- | :--- | :--- | :--- |
| **forge 數據集** | 26 筆資料<br>2 個特徵 (2D)<br>2 個類別 | **最直觀的二元分類**：特徵只有 2 個，最適合直接在平面散佈圖上畫出分類邊界。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E6%95%B8%E6%93%9A%E5%9F%BA%E7%A4%8E%E8%88%87NumPy%E5%AF%A6%E6%88%B0/forge%E6%95%B8%E6%93%9A%E9%9B%86.ipynb)<br>[開啟 forge數據集.ipynb](./forge數據集.ipynb) |
| **Iris 鳶尾花數據集** | 150 筆資料<br>4 個特徵<br>3 個花種品系 | **經典入門教材**：學習用 NumPy 進行資料切片、篩選特定花種、繪製多特徵分佈散佈圖。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E6%95%B8%E6%93%9A%E5%9F%BA%E7%A4%8E%E8%88%87NumPy%E5%AF%A6%E6%88%B0/Iris%E6%95%B8%E6%93%9A%E9%9B%86%E9%80%B2%E8%A1%8C%E6%95%B8%E6%93%9A%E9%81%B8%E6%93%87%E8%88%87%E5%88%87%E7%89%87.ipynb)<br>[開啟 Iris數據集.ipynb](./Iris數據集進行數據選擇與切片.ipynb) |
| **威斯康辛州乳癌數據集** | 569 筆資料<br>30 個特徵<br>良性 / 惡性 | **真實高維醫療診斷**：體驗當特徵多達 30 個時，如何觀察各維度的尺度差異與統計特性。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E6%95%B8%E6%93%9A%E5%9F%BA%E7%A4%8E%E8%88%87NumPy%E5%AF%A6%E6%88%B0/%E5%A8%81%E6%96%AF%E5%BA%B7%E8%BE%9B%E5%B7%9E%E4%B9%B3%E7%99%8C%E6%95%B8%E6%93%9A%E9%9B%86_load_breast_cancer.ipynb)<br>[開啟 乳癌數據集.ipynb](./威斯康辛州乳癌數據集_load_breast_cancer.ipynb) |

---

### 📈 任務類型 B：迴歸問題 (Regression)
> **目標**：讓演算法預測一組連續的目標數值。

| 數據集名稱 | 資料規模與維度 | 教學特色與重點 | 線上執行與檔案 |
| :--- | :--- | :--- | :--- |
| **wave 數據集** | 40 筆資料<br>1 個輸入特徵<br>1 個連續目標 | **最易懂的單變量迴歸**：1 個 X 對應 1 個 y，最適合用來視覺化回歸線擬合過程。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E6%95%B8%E6%93%9A%E5%9F%BA%E7%A4%8E%E8%88%87NumPy%E5%AF%A6%E6%88%B0/wave%E6%95%B8%E6%93%9A%E9%9B%86.ipynb)<br>[開啟 wave數據集.ipynb](./wave數據集.ipynb) |
| **加州房價數據集** | 20,640 筆資料<br>8 個特徵<br>預測房價中位數 | **真實大型迴歸問題**：包含區域收入、平均屋齡等特徵，體驗真實複雜房價預測。 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roberthsu2003/machine_learning/blob/main/%E6%95%B8%E6%93%9A%E5%9F%BA%E7%A4%8E%E8%88%87NumPy%E5%AF%A6%E6%88%B0/%E5%8A%A0%E5%B7%9E%E6%88%BF%E5%83%B9%E6%95%B8%E6%93%9A%E9%9B%86_fetch_california_housing.ipynb)<br>[開啟 加州房價.ipynb](./加州房價數據集_fetch_california_housing.ipynb) |
| **Salary 年資薪水數據集** | 30 筆資料<br>年資 vs. 薪水 (CSV 格式) | **外部檔案實戰**：練習以 Pandas 載入本地 CSV 檔案，並轉換為 NumPy 陣列進行分析。 | [檢視 Salary_Data.csv](./Salary_Data.csv) |

---

## 🌐 5. 更多公開數據集資源（自學與專案探索庫）

學會基本套件與數據操作後，如果想自主尋找有趣的題目，推薦以下兩個國際最知名的資料庫：

* 🏛️ **[UCI 機器學習資料庫 (UCI Machine Learning Repository)](https://archive.ics.uci.edu/datasets)**：  
  加州大學歐文分校（UCI）建立的經典開源資料庫，提供數百個依照演算法任務（Classification, Regression, Clustering）與資料型態分類好的標竿數據集。
* 🏆 **[Kaggle Datasets 開源數據專區](https://www.kaggle.com/datasets)**：  
  全球最大的資料科學家與機器學習社群，擁有各行各業的海量數據（金融、醫療、行銷、影像、文字等），並附帶全球開發者公開分享的 Jupyter Notebook 實作分析範例。
