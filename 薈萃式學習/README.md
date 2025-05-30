# 薈萃式學習-集成學習(Ensemble Learning)

薈萃式學習（集成學習）是一種機器學習方法，通過結合多個基礎模型（稱為弱學習器或基學習器）的預測結果來提高整體模型的性能。這些基礎模型可以是決策樹、線性模型、神經網絡等，通過特定的組合策略（如投票、平均或加權）生成最終預測。薈萃式學習的核心思想是**「集體智慧」**，即多個模型的協同工作通常比單一模型更穩健且準確。

### 常見的薈萃式學習方法包括：
- **Bagging（Bootstrap Aggregating）**：如隨機森林（Random Forest），通過對數據進行有放回採樣訓練多個模型，並平均或投票得出結果。

> [!TIP]
> [**Bagging**(Boostrap Aggregating)的細節說明](./Bagging(Bootstrap_Aggregating)說明.md)

- **Boosting**：如AdaBoost、Gradient Boosting和XGBoost，通過迭代地調整樣本權重或梯度來訓練模型，逐步提升性能。

> [!TIP]
> [**Boosting**的細節說明](./Boosting說明.md)

- **Stacking**：將多個不同類型的模型預測結果作為輸入，訓練一個元模型（meta-model）進行最終預測。

> [!TIP]
> [**Staking**的細節說明](./Stacking說明.md)


### 薈萃式學習的優點:

1. **提高預測準確性:** 通過結合多個模型的預測，減少單一模型的錯誤，提升整體性能。
2. **降低過擬合風險:** Bagging方法（如隨機森林）通過隨機採樣和特征選擇，減少模型對訓練數據的過度依賴。

3. **增強模型穩定性:** 薈萃式學習對數據中的噪音和異常值不敏感，能在不同數據分佈下保持穩定。

4. **靈活性:** 可結合不同類型的基礎模型，適應多種問題和數據類型。

5. **特征重要性評估:** 如隨機森林和Gradient Boosting能提供特征重要性分數，幫助理解數據中的關鍵變量。

### 範例:直觀理解薈萃式學習的優勢

- 預測模型(regression)

> [!IMPORTANT]
> - 比較單一決策樹與隨機森林（一種薈萃式學習方法）  
> - 不同數據集大小下的預測誤差。  
> - 使用Matplotlib生成一張圖表  
> - [直觀理解薈萃式學習的優勢_regression-ipynb範例](./emsemble_直觀理解薈萃式學習的優勢.ipynb)

![](./ensemble_vs_single_model_real_training.png)

### 範例:

- 分類模型(classification)

> [!IMPORTANT]
> - 隨機森林的分類範例
> - 何應用薈萃式學習解決簡單的二元分類問題。
> - 該範例生成一個包含1000個樣本和20個特征的模擬數據集。
> - 使用隨機森林（100棵決策樹）進行二元分類。
> - 輸出模型在測試集上的準確率，展示薈萃式學習的應用。
> - 隨機森林的簡單實現，並解釋其內部如何通過多棵決策樹的投票提高預測性能。
> - [直觀理解薈萃式學習的優勢_classification-ipynb範例](./ensemble_直觀理解薈萃式學習的優勢1.ipynb)

![](./ensemble_vs_single_model_classification_accuracy.png)


### Bagging實際案例

金融風險評估-使用UCI Credit Card Default Dataset進行違約風險預測：
	
在金融風險評估領域，薈萃式學習（例如隨機森林、Gradient Boosting）是解決客戶違約預測、信用評分及詐欺偵測等任務的強效工具。這些方法藉由整合多個基礎學習器（弱學習器）的預測，不僅能更精準地應對複雜的金融數據，同時也顯著提升了模型在面對數據噪音和樣本不平衡情況時的穩健性。

公開數據集來源

以下是兩個適合金融風險評估的公開數據集，這些數據集可在網路上免費取得：

1. UCI Credit Card Default Dataset：
	- 來源: UCI Machine Learning Repository（https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients）

	- 內容: 該數據集包含台灣某銀行的信用卡客戶數據（2005年4月至9月），包括23個特征（如信用額度、性別、教育程度、還款記錄等）以及是否違約的標籤（二元分類：違約/未違約）。

	- 樣本數：30,000筆記錄，適合用於違約風險預測。

	- 適用性：模擬真實的信用卡違約風險評估場景。

以下是一個使用Python和Scikit-learn實現的範例，展示如何使用隨機森林（一種ensemble學習方法）對信用卡違約風險進行預測，並比較其與單一決策樹的性能。該範例包括數據下載、預處理、模型訓練和圖表生成，以幫助學生理解薈萃式學習的優勢。

> [!IMPORTANT]
> 金融風險評估.  
> [ensemble_金融風險評估-ipynb範例](./ensemble_金融風險評估.ipynb)

![](./model_performance_comparison.png)


### 擴展練習

- 嘗試調整隨機森林的參數（如樹的數量n_estimators或最大深度max_depth），觀察對性能的影響；或使用Gradient Boosting（如XGBoost）進行比較。

---

### Boosting 實際案例
**Boosting** 演算法的範例，

**UCI 心臟病數據集 (Heart Disease UCI)**。

> [!TIP]
> **這個範例將包含：**
> 1. 載入 UCI 心臟病數據集。
> 2. 進行數據預處理，包括：
>    - 賦予欄位名稱。
>    - 處理數據集中的缺失值（通常以 '?' 表示）。
>    - 對類別型特徵進行獨熱編碼 (One-Hot Encoding)。
>    - 對數值型特徵進行標準化 (Standard Scaling)。
> 3. 訓練一個基礎模型（例如：決策樹）作為比較基準。
> 4. 訓練一個 Boosting 模型（我們將使用 `GradientBoostingClassifier`）。
> 5. 評估兩個模型的準確率 (Accuracy) 和 ROC-AUC 分數。
> 6. 繪製一個長條圖，比較這些模型的效能。
> 7. 儲存圖表並顯示。  
> [**集成學習Boosting-ipynb範例**](./ensemble_Boosting範例.ipynb)

![Boosting 實際案例](./heart_disease_boosting_performance_comparison.png)


---

### Stacking 實際案例
我將為您提供一個使用 Pima 印第安人糖尿病數據集 (Pima Indians Diabetes Database) 的 Stacking 學習範例，並將其整合到一個儲存格中，包含數據載入、預處理、Stacking 模型訓練、基礎模型訓練（用於比較）、評估以及效能比較圖表的繪製與儲存。這個數據集來自 UCI 機器學習數據庫，包含了 Pima 印第安女性的幾項醫療預測變量以及她們是否在五年內患上糖尿病的資訊。這是一個典型的二元分類問題。

> [!TIP]
> **這個範例會：**
> 1. 載入 Pima 印第安人糖尿病數據集。
> 2. 進行必要的數據預處理（處理不合理的零值、特徵標準化）。
> 3. 定義幾個基礎學習器（例如：邏輯回歸、決策樹、支持向量機）。
> 4. 定義一個元學習器（例如：邏輯回歸）。
> 5. 訓練 Stacking 分類器。
> 6. 為了比較，也會單獨訓練和評估這些基礎學習器。
> 7. 計算所有模型（基礎學習器和 Stacking 分類器）的準確率和 ROC-AUC 分數。
> 8. 繪製一個長條圖，比較這些模型的效能。
> 9. 儲存圖表並顯示。  
> [**集成學習Stacking-ipynb範例**](./Stacking範例.ipynb)

![tacking實作](./pima_diabetes_stacking_performance_comparison.png)