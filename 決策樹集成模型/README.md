# 🌲 決策樹集成模型 (Ensembles of Decision Trees) 學習與實作指南

歡迎來到機器學習 **決策樹集成模型（Ensembles of Decision Trees）** 的教學目錄。本單元深入探討如何透過組合多棵決策樹來克服單一決策樹容易過擬合（Overfitting）的弱點，包含 **隨機森林（Random Forests）** 與 **梯度提升樹（Gradient Boosted Decision Trees / GBDT）** 的核心理論、邊界視覺化、特徵重要性分析與調參指南。

---

## 📌 🎯 ⭐ 主教學筆記本（Master Notebook）

> [!IMPORTANT]
> 💡 **首選推薦**：請優先開啟 [randomForestEnsemble.ipynb](./randomForestEnsemble.ipynb)！
> 
> 此檔案為本章節最完整、最詳盡的**主教學筆記本**。內含 Bagging / Boosting / Stacking 理論比較、5 棵樹與集成邊界繪製 (圖 2-33)、乳癌資料集 100 棵樹隨機森林 (圖 2-34)、梯度提升樹 (GBDT) 實作與調參 (圖 2-35) 以及完整的實務調參指南！

---

## 📂 檔案架構與規劃說明 (Directory Overview & Renaming Roadmap)

本資料夾已參照《樹狀模型》章節的架構進行重構，將所有範例模組化為結構清晰的筆記本：

| 檔名 / 目錄 | 說明與學習主題 | 建議用途 / 對應章節 |
| :--- | :--- | :--- |
| 📌 **[randomForestEnsemble.ipynb](./randomForestEnsemble.ipynb)** | **主教學核心筆記本**（全觀念整合、全繁體中文、含完整圖文對照與 GBDT） | 📖 完整集成模型學習 / 教學展示 |
| 📄 **[01_random_forest_5trees.ipynb](./01_random_forest_5trees.ipynb)** *(原 分析Random_Forests.ipynb)* | 5 棵決策樹邊界與隨機森林整體邊界視覺化 | 🔍 觀察多樹合體邊界平滑化 |
| 📄 **[02_random_forest_cancer.ipynb](./02_random_forest_cancer.ipynb)** *(原 100棵樹組成的隨機森林.ipynb)* | 乳癌資料集 - 100 棵樹隨機森林與特徵重要性直方圖 | 🌲 隨機森林抗過擬合實作 |
| 🚀 **[03_gradient_boosting_cancer.ipynb](./03_gradient_boosting_cancer.ipynb)** *(全新補充)* | 乳癌資料集 - 梯度提升樹 (GBDT) 實作、`max_depth` 與 `learning_rate` 調參 | ⚡ GBDT 循序學習與優化 |
| 📄 **[04_synthetic_rf_demo.ipynb](./04_synthetic_rf_demo.ipynb)** *(原 random_forests1.ipynb)* | 合成資料集二元分類實作與單一樹結構繪製 | 🧪 基礎實作練習 |

---

## 🗺️ 決策樹集成學習地圖與圖表對照索引 (Learning Roadmap)

### 1. 集成模型三大主要類型 (Bagging, Boosting, Stacking)
* **概念**：
  * **Bagging (裝袋法 / 代表：隨機森林)**：同時平行建立多棵獨立樹，最後投票或平均。
  * **Boosting (提升法 / 代表：GBDT, XGBoost)**：一棵一棵循序建立樹，專注於修正前樹殘差與錯誤。
  * **Stacking (堆疊法)**：結合不同演算法的第一層預測，作為第二層最終模型的特徵。
* **對應單元**：[randomForestEnsemble.ipynb (Cell 01)](./randomForestEnsemble.ipynb)

### 2. 隨機森林原理與邊界平滑化
* **概念**：透過 **Bootstrap 抽樣** 與 **`max_features` 特徵子集隨機化**，確保個體樹的差異性，抵銷過擬合。
* **對應圖表**：📌 **圖 2-33**（5 棵樹獨立決策邊界與隨機森林整體合成邊界）
* **對應單元**：[01_random_forest_5trees.ipynb](./01_random_forest_5trees.ipynb)

### 3. 隨機森林特徵重要性 (Feature Importances)
* **概念**：隨機森林計算的特徵重要性透過多樹平均，比單一決策樹更平滑、更可靠（考量了更多特徵而不會過度依賴單一特徵）。
* **對應圖表**：📌 **圖 2-34**（100 棵樹隨機森林特徵重要性直方圖）
* **對應單元**：[02_random_forest_cancer.ipynb](./02_random_forest_cancer.ipynb)

### 4. 梯度提升樹 (GBDT) 實作與調參
* **概念**：使用深度較淺的樹（`max_depth=1`）與低學習率（`learning_rate=0.01`）避免過擬合，獲得高度精準的預測力。
* **對應圖表**：📌 **圖 2-35**（GBDT 特徵重要性分佈圖）
* **對應單元**：[03_gradient_boosting_cancer.ipynb](./03_gradient_boosting_cancer.ipynb)

---

## 🛠️ 依賴套件需求 (Requirements)

在執行各 Notebook 前，請確保 Python 環境已安裝下列套件：

```bash
pip install numpy pandas matplotlib scikit-learn mglearn
```