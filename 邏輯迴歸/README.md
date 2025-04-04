## 邏輯分類(Logistic Regression)
邏輯迴歸（Logistic Regression），這是一個在機器學習中非常重要且應用廣泛的分類演算法

### 重點
- 解釋邏輯迴歸的基本概念，以及它如何用於二元分類。
- 解釋迴歸係數和截距的意義，以及它們如何影響預測結果。
- 嘗試修改數據集，觀察模型的變化。
- 介紹其他評估指標，例如精確度、召回率和 F1 分數，來評估模型的性能。
- 使用真實世界的數據集，例如患者數據集，來進行邏輯迴歸分析。
- 解釋邏輯回歸與線性回歸的差異。
- 解釋Sigmoid函數在邏輯回歸扮演的角色。
- 解釋如何使用機率來解釋邏輯回歸的輸出。



### 什麼是邏輯迴歸？

* **分類問題：**
    * 邏輯迴歸主要用於解決分類問題，也就是預測一個事件屬於哪個類別。例如，預測一封郵件是否為垃圾郵件、判斷一位病人是否患有某種疾病等。
* **預測機率：**
    * 與線性迴歸不同，邏輯迴歸的輸出不是連續的數值，而是介於 0 和 1 之間的機率值，表示某事件發生的可能性。
* **Sigmoid 函數：**
    * 邏輯迴歸使用 Sigmoid 函數（或稱為 Logistic 函數）將線性迴歸的輸出轉換為機率值。這個函數的特點是將任何實數映射到 0 到 1 之間。

### 邏輯迴歸的運作原理

1.  **線性組合：**
    * 首先，邏輯迴歸像線性迴歸一樣，計算輸入特徵的加權總和。
2.  **Sigmoid 轉換：**
    * 然後，將這個線性組合的結果輸入 Sigmoid 函數，得到一個介於 0 和 1 之間的機率值。
3.  **閾值判斷：**
    * 最後，設定一個閾值（通常為 0.5），如果機率值大於閾值，則預測為某一類別；否則，預測為另一類別。

### 邏輯迴歸的應用場景

* **醫學診斷：**
    * 預測病人是否患有某種疾病。
* **信用評估：**
    * 判斷客戶是否具有信用風險。
* **垃圾郵件檢測：**
    * 識別郵件是否為垃圾郵件。
* **市場行銷：**
    * 預測客戶是否會購買某種產品。

### 邏輯迴歸的優點

* **簡單且易於實現：**
    * 邏輯迴歸的數學原理相對簡單，容易理解和實現。
* **輸出機率值：**
    * 可以輸出事件發生的機率，這在許多應用場景中非常有用。
* **適用於二元分類：**
    * 特別適合於二元分類問題。

### 邏輯迴歸的限制

* **線性可分性假設：**
    * 邏輯迴歸假設資料是線性可分的，如果資料不是線性可分的，則效果可能不佳。
* **對特徵工程的要求：**
    * 邏輯迴歸的性能很大程度上取決於特徵工程的質量。

### 實作
> [!IMPORTANT]
> [**Logistic Regression VS LinearSVC**](./說明1.ipynb)  
> [**真偽資料集**實作解說](./forge說明2.ipynb)  
> [**斯康辛州乳癌資料集**實作解說](./cancer說明3.ipynb)  
> [**多元線性分類說明**](./multiclass_classification說明.ipynb)  
> [**多元線性分類實作**](./multiclass_classification實作.ipynb)  
 





