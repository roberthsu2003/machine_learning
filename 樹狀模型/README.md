# 🌳 樹狀模型 (Tree-based Models) 學習與實作指南

歡迎來到機器學習 **樹狀模型（Tree-based Models）** 的教學目錄。本單元包含從基礎概念比喻、漸進切割繪圖、過擬合與預剪枝實作、Graphviz 可視化、特徵重要性分析，到迴歸外推限制與實務專案演練的完整教學教材。

---

## 📌 🎯 ⭐ 主教學筆記本（Master Notebook）

> [!IMPORTANT]
> 💡 **首選推薦**：請優先開啟 [decisionThreeClassifier.ipynb](./decisionThreeClassifier.ipynb)！
> 
> 此檔案為本章節最完整、最詳盡的**主教學筆記本**。內含完整的決策樹理論觀念、漸進圖像繪製 (depth=1, 2, 9)、乳癌資料集預剪枝實作、Graphviz 樹狀圖解讀、特徵重要性直方圖與決策樹迴歸外推能力（Extrapolation）測試！

---

## 📂 檔案架構與規劃說明 (Directory Overview & Renaming Roadmap)

為方便教學與分頁學習，本資料夾已將原本命名不夠明確的範例檔（原 `demo1` ~ `demo5`）重構為模組化筆記本：

| 檔名 / 目錄 | 說明與學習主題 | 建議用途 / 對應章節 |
| :--- | :--- | :--- |
| 📌 **[decisionThreeClassifier.ipynb](./decisionThreeClassifier.ipynb)** | **主教學核心筆記本**（全觀念整合、全繁體中文、含完整圖文對照與迴歸外推） | 📖 完整課程學習 / 教學展示 |
| 📄 **[01_animal_tree.ipynb](./01_animal_tree.ipynb)** *(原 demo1.ipynb)* | 「二十問 (Twenty Questions)」動物分類決策樹範例 | 🐣 決策樹基礎概念入門 |
| 📄 **[02_unpruned_tree.ipynb](./02_unpruned_tree.ipynb)** *(原 demo2.ipynb)* | 乳癌資料集 - 未剪枝決策樹實作與過擬合 (Overfitting) 分析 | ⚠️ 觀察純葉節點過擬合現象 |
| 📄 **[03_prepruning_tree.ipynb](./03_prepruning_tree.ipynb)** *(原 demo3.ipynb)* | 乳癌資料集 - 預先剪枝 (`max_depth=4`) 提升泛化能力 | ✂️ 預剪枝策略實作 |
| 📄 **[04_tree_visualization.ipynb](./04_tree_visualization.ipynb)** *(原 demo4.ipynb)* | 導出 Graphviz `.dot` 檔與決策樹結構圖繪製 | 🔍 樹狀圖節點結構解讀 |
| 📄 **[05_feature_importance.ipynb](./05_feature_importance.ipynb)** *(原 demo5.ipynb)* | 決策樹特徵重要性 (`feature_importances_`) 計算與條形圖 | 📊 特徵貢獻度分析 |
| 🌸 **[sklearn實作1.ipynb](./sklearn實作1.ipynb)** | Scikit-Learn `DecisionTreeClassifier` 經典 Iris 鳶尾花分類實作 | 🧪 獨立手把手練習 |
| 📚 **[sklearn實作2/](./sklearn實作2/)** | **書店文具銷售數據分類實作專案**（包含完整數據集、增強型腳本與分析圖表） | 💼 實務專案應用 |

---

## 🗺️ 決策樹學習地圖與圖表對照索引 (Learning Roadmap)

### 1. 決策樹基礎與問答邏輯 (Twenty Questions)
* **概念**：決策樹透過學習一系列 if/else 問題層級結構來得出分類結論（如是否有羽毛、是否會飛、是否有鰭區分熊、鷹、企鵝、海豚）。
* **對應單元**：[01_animal_tree.ipynb](./01_animal_tree.ipynb)

### 2. 空間劃分與深度 (Depth) 解析
* **概念**：以 `two_moons` 雙月資料集示範，隨著深度 (`max_depth=1, 2, 9`) 增加，空間分割與樹狀結構的漸進變化。
* **對應圖表**：
  * 📌 **圖 2-23**：`two_moons` 資料集散佈圖
  * 📌 **圖 2-24 ~ 圖 2-26**：`depth=1, 2, 9` 邊界切分與過擬合現象
* **對應單元**：[decisionThreeClassifier.ipynb (Cell 03~06)](./decisionThreeClassifier.ipynb)

### 3. 過擬合與預先剪枝 (Pre-pruning)
* **概念**：完全生長的未剪枝樹（訓練集 100% 準確率）極易產生過擬合；透過設定 `max_depth=4` 預先剪枝，可有效控制模型複雜度並顯著提升測試集準確率。
* **對應單元**：[02_unpruned_tree.ipynb](./02_unpruned_tree.ipynb) 與 [03_prepruning_tree.ipynb](./03_prepruning_tree.ipynb)

### 4. 樹狀圖可視化與結構解讀
* **概念**：使用 `export_graphviz` 導出 `.dot` 檔並繪製決策樹結構圖，拆解 `samples`（樣本數）、`value`（類別分佈）與 `class`（判定類別）。
* **對應圖表**：📌 **圖 2-27**（乳癌資料集決策樹結構圖）
* **對應單元**：[04_tree_visualization.ipynb](./04_tree_visualization.ipynb)

### 5. 特徵重要性分析 (Feature Importances)
* **概念**：計算特徵權重評分（介於 0 到 1 之間，總和為 1），觀察「最大半徑 (worst radius)」對決策的重大貢獻。
* **對應圖表**：
  * 📌 **圖 2-28**：乳癌特徵重要性繁體中文水平條形圖
  * 📌 **圖 2-29 & 圖 2-30**：非單調特徵與決策邊界圖
* **對應單元**：[05_feature_importance.ipynb](./05_feature_importance.ipynb)

### 6. 決策樹迴歸與無法外推特性 (Extrapolation)
* **概念**：在歷史 RAM 價格資料集 (`ram_price.csv`) 上比較 `DecisionTreeRegressor` 與 `LinearRegression`；說明決策樹因分割區域限制而無法預測訓練集範圍外未來趨勢的理由。
* **對應圖表**：📌 **圖 2-31**（RAM 價格歷史趨勢與外推預測比較圖）
* **對應單元**：[decisionThreeClassifier.ipynb (Cell 22~26)](./decisionThreeClassifier.ipynb)

---

## 🛠️ 依賴套件需求 (Requirements)

在執行各 Notebook 前，請確保 Python 環境已安裝下列套件：

```bash
pip install numpy pandas matplotlib scikit-learn mglearn graphviz
```

> ⚠️ **系統說明**：若要在本地端渲染 Graphviz `.dot` 圖像，需確保系統已安裝 Graphviz 執行檔（例如 Mac 可透過 `brew install graphviz` 安裝）。