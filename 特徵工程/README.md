# 特徵工程 (Feature Engineering)

機器學習中一個至關重要的步驟：特徵工程。良好的特徵工程可以顯著提升模型的性能，而差的特徵工程則可能導致模型無法學習到有用的資訊。

## 什麼是特徵工程？

特徵工程是利用領域知識來建立特徵，讓機器學習演算法得以運作的過程。它包含了特徵的創造、轉換、提取和選擇。

## 講義大綱

1.  **數據預處理 (Data Preprocessing)**
    在訓練模型之前，清理數據並處理缺失值以確保數據品質。
    *   [處理缺失值](./examples/1_handling_missing_values.ipynb)：刪除或以平均值、最頻繁值等方式填補數據中的缺失欄位。
    *   [數據清洗](./examples/2_data_cleaning.ipynb)：處理重複數據、修正格式或欄位值的類別不一致性。
2.  **處理類別數據 (Handling Categorical Data)**
    將非數值的類別特徵轉換為機器學習模型能夠處理的數值格式。
    *   [標籤編碼 (Label Encoding)](./examples/3_label_encoding.ipynb)：將類別資料直接映射為遞增的整數編碼（適用於有序數據）。
    *   [獨熱編碼 (One-Hot Encoding)](./examples/4_one_hot_encoding.ipynb)：為每個類別建立獨立的二元特徵（0 或 1），避免引入多餘的順序關係。
3.  **特徵縮放 (Feature Scaling)**
    將不同範圍的數值特徵調整至一致的尺度，避免某些大範圍特徵主導模型的學習。
    *   [標準化 (Standardization)](./examples/5_standardization.ipynb)：將特徵數據轉換為平均值為 0、標準差為 1 的常態分佈。
    *   [歸一化 (Normalization)](./examples/6_normalization.ipynb)：將特徵數據等比例縮放到指定的範圍（通常是 0 到 1 之間）。
4.  **特徵創造 (Feature Creation)**
    基於現有特徵創造出新的特徵，以幫助模型捕捉更複雜的非線性關係或協同效應。
    *   [多項式特徵 (Polynomial Features)](./examples/7_polynomial_features.ipynb)：藉由升冪（如二次項）建立多項式特徵，捕捉非線性模式。
    *   [互動特徵 (Interaction Features)](./examples/8_interaction_features.ipynb)：計算特徵之間的乘積項，藉此捕捉多個特徵間的交互協同效應。
5.  **特徵選擇 (Feature Selection)**
    從所有特徵中挑選出對模型預測最有關聯、最有效率的特徵子集，以降低模型複雜度並防範過擬合。
    *   [過濾法 (Filter Methods)](./examples/9_filter_methods.ipynb)：利用統計指標（如方差、相關係數、F-value）對特徵單獨評分進行篩選。
    *   [包裝法 (Wrapper Methods)](./examples/10_wrapper_methods.ipynb)：將特徵選擇視為搜尋問題，利用機器學習模型遞歸消除不重要的特徵（如 RFE）。
    *   [嵌入法 (Embedded Methods)](./examples/11_embedded_methods.ipynb)：在模型訓練過程中自動完成特徵篩選（如 Lasso 正規化或隨機森林特徵重要性）。
6.  **降維 (Dimensionality Reduction)**
    在盡可能保留原始數據特徵與變異數的前提下，減少特徵的維度空間。
    *   [主成分分析 (Principal Component Analysis, PCA)](./examples/12_pca.ipynb)：最常用的線性降維方法，將高度相關的原始特徵投影至互相垂直的低維主成分空間。

---

