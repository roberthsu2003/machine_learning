# 1. 機器學習的類型 (Types of Machine Learning)

<!-- interactive-glossary:start -->

🌐 **[開啟本章互動動畫教學](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/)**

動畫場景：[監督式：分類與回歸](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/#supervised) · [非監督式：聚類與降維](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/#unsupervised) · [學習類型比較](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/#paradigm-comparison)

<!-- interactive-glossary:end -->

機器學習（Machine Learning, ML）本質上是**讓電腦從數據中自動尋找規律，而非依賴人工程式碼一條條寫死規則**的科學。

依據「是否有標準答案提供給模型學習」以及「回饋機制的不同」，機器學習主要分為兩大主要類型：**監督式學習**與**非監督式學習**；在進階實務中，亦常結合**半監督式學習**與**強化學習**。

---

## 🎯 本章學習目標
1. 掌握**監督式學習**的核心機制，能精準區分**分類任務**與**回歸任務**。
2. 理解**非監督式學習**的探索本質，掌握**聚類**與**降維**的應用場景。
3. 能從資料特性（有無標籤）與商業需求出發，正確選定合適的學習類型。

---

## 1. 監督式學習 (Supervised Learning)

### 核心定義
監督式學習是指使用**帶有標籤（Ground Truth Label）**的訓練數據來訓練模型。模型透過學習輸入特徵 $X$ 與對應標籤 $y$ 之間的對應映射關係 $f(X) \approx y$，進而能夠對從未見過的新數據做出準確預測。

### 💡 生活直觀比喻：名師出高徒與刷題附詳解
> 監督式學習就像一位學生在做**附有詳細解答與正確答案的歷屆考題本**。  
> 學生每做完一題，就翻到最後一頁對答案（計算誤差 Loss），答錯了就修正自己的解題思考（調整權重參數），直到錯誤率降到最低為止。

#### 📊 觀念圖解
> 🎬 互動動畫：[監督式：分類與回歸](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/#supervised)
![監督式學習圖解](./images/01_supervised_learning.png)

### 兩大主要任務類型
監督式學習依據預測目標的數據型態，分為兩大主要任務：

| 任務類型 | 標籤型態 | 預測目標特點 | 經典生活實例 | 常用代表算法 |
| :--- | :--- | :--- | :--- | :--- |
| **分類 (Classification)** | 離散類別 (Discrete) | 預測數據屬於哪一個類別標籤（二分類或多分類） | • 判斷信件是否為垃圾郵件<br>• 醫療影像診斷是否罹癌<br>• 信用卡交易是否為盜刷 | • 邏輯回歸 (Logistic Regression)<br>• 決策樹 (Decision Tree)<br>• 支援向量機 (SVM)<br>• 隨機森林 (Random Forest) |
| **回歸 (Regression)** | 連續數值 (Continuous) | 預測一個可以在數線上連續變動的具體數量值 | • 根據地段坪數預測房價<br>• 預測明日氣溫與降雨量<br>• 預測下季度公司銷售額 | • 線性回歸 (Linear Regression)<br>• 脊回歸 / Lasso 回歸<br>• 回歸樹 / GBDT<br>• 神經網絡 (Neural Networks) |

### 經典範例解說
假設我們要預測房屋價格：
- **輸入特徵 $X$**：建物面積 = 100 平方公尺，房間數 = 2，距捷運站距離 = 300 公尺。
- **目標標籤 $y$**：成交價格 = 1,800 萬元。
- **模型任務**：找出 $X$ 與 $y$ 之間的函數映射。當市場出現一間全新房屋（面積 120 平方公尺，房間數 3）時，模型能精準推估其合理售價。

---

## 2. 非監督式學習 (Unsupervised Learning)

### 核心定義
非監督式學習是指在**完全沒有標籤（無標準答案）**的數據上進行學習。模型不依賴外界給予的正確答案，而是透過自主統計與幾何計算，發掘數據內部潛藏的內在結構、相似族群或低維流形模式。

### 💡 生活直觀比喻：讓小朋友分類彩色積木
> 想像把一大箱形狀各異、五顏六色的積木倒在地上，**不給任何指示或說明書**，讓小朋友自己整理。  
> 小朋友會憑直覺把「圓形的積木放在一起」、「紅色的積木分到同一堆」。雖然他並不知道這些積木的專有名詞，但他成功抓住了積木之間的「相似性特徵」。

#### 📊 觀念圖解
> 🎬 互動動畫：[非監督式：聚類與降維](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/#unsupervised)
![非監督式學習圖解](./images/02_unsupervised_learning.png)

### 兩大主要任務類型

| 任務類型 | 核心目標 | 實務價值 | 經典實例 | 代表算法 |
| :--- | :--- | :--- | :--- | :--- |
| **聚類 (Clustering)** | 將相似的數據點自動聚集為同一群，群內相似度高、群間差異大 | 探索未知的用戶分群或數據分佈 | • 電商顧客消費偏好分群<br>• 新聞主題自動分堆<br>• 基因表達譜分型 | • K-Means 聚類<br>• DBSCAN 密度聚類<br>• 階層聚類 (Hierarchical) |
| **降維 (Dimensionality Reduction)** | 在盡可能保留原始資訊的前提下，壓縮特徵維度 | 解決維度災難、消除特徵冗餘、資料視覺化 | • 將 100 維的基因特徵壓縮為 2 維以繪圖呈現<br>• 圖像壓縮與去噪 | • 主成分分析 (PCA)<br>• t-SNE / UMAP<br>• 自編碼器 (Autoencoder) |

---

## 3. 學習類型全景對照

#### 📊 觀念圖解
> 🎬 互動動畫：[學習類型比較](https://roberthsu2003.github.io/machine_learning/glossary/01-learning-paradigms/#paradigm-comparison)
![學習類型總結比較圖解](./images/03_comparison.png)

### 核心對照矩陣

| 比較維度 | 監督式學習 (Supervised) | 非監督式學習 (Unsupervised) |
| :--- | :--- | :--- |
| **訓練數據組成** | 特徵矩陣 $X$ + **標準標籤 $y$** | 僅有特徵矩陣 $X$（**無標籤**） |
| **學習目標** | 預測未知樣本的輸出值（求對應） | 發現數據自身的結構、分佈與族群（求規律） |
| **人工作業成本** | **高**（需要大量人工標註數據） | **低**（可直接利用未經整理的原始數據） |
| **模型評估方式** | 客觀明確（準確率、召回率、MSE 等） | 較具主觀性（輪廓係數、重構誤差、領域專家解讀） |
| **常見應用領域** | 疾病診斷、人臉識別、股價走勢預測 | 潛在客群輪廓分析、異常詐欺交易偵測、圖像特徵壓縮 |

> [!NOTE]
> **拓展視野：其他現代學習類型**
> - **半監督式學習 (Semi-Supervised Learning)**：擁有大量未標記數據，但只有極少量昂貴的人工標籤。利用少量標籤引導大量未標籤數據，大幅降低標註成本。
> - **強化學習 (Reinforcement Learning, RL)**：沒有固定的數據集，智慧體（Agent）透過與環境互動試錯（Trial-and-Error），依據獲得的獎勵（Reward）或懲罰最大化長期累積回報（如 AlphaGo、自動駕駛）。

---

## 📌 本章精華速記
1. **監督式學習**看重「對齊答案」，核心是輸入到輸出的對應映射，分為**預測類別的分類**與**預測數值的回歸**。
2. **非監督式學習**看重「自發探索」，核心是資料內在結構的解析，主要代表為**分群聚類**與**特徵降維**。
3. 在真實機器學習專案中，通常先以非監督式方法（如 PCA、聚類）進行探索性數據分析（EDA），再透過監督式模型執行精準預測。

---

[📑 返回目錄](../README.md) | [⏭️ 下一章：02_數據結構](../02_數據結構/README.md)
