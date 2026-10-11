# 🚀 機器學習模型部署專案目錄 (Machine Learning Model Deployment)

本目錄收錄了將機器學習模型從演算法訓練走向雲端 Web 服務的完整實戰專案，架構均採用 **FastAPI (高效 RESTful API)** 與 **Gradio (低代碼互動 Web UI)** 混合掛載技術。

---

## 📂 專案實作清單

| 專案名稱 | 任務領域 | 必備先修演算法與相關資料夾 | 核心演算法教學章節 | 部署步驟拆解 Notebook | 專案講義與程式碼 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **薪資預測** | 商業分析 | 📐 [簡單線性迴歸](../簡單線性迴歸/README.md)<br>📊 [多元線性迴歸](../多元線性迴歸/README.md) | 📘 [multiple_linear_regression2.ipynb](../多元線性迴歸/multiple_linear_regression2.ipynb) | 🎓 [train API](./薪資預測-多元線性迴歸/practice_train_api_answer.ipynb)<br>🎓 [predict API](./薪資預測-多元線性迴歸/practice_predict_api_answer.ipynb)<br>🎓 [gradio UI](./薪資預測-多元線性迴歸/practice_gradio_ui_answer.ipynb) | 📊 [專案講義](./薪資預測-多元線性迴歸/README.md) |
| **鳶尾花品種分類** | 生物形態 | 🌿 [樹狀模型 (單一決策樹)](../樹狀模型/README.md)<br>🌲 [決策樹集成模型 (隨機森林)](../決策樹集成模型/README.md) | 🌲 [01_random_forest_5trees.ipynb](../決策樹集成模型/01_random_forest_5trees.ipynb) | 🎓 [train API](./鳶尾花隨機森林分類器/practice_train_api_answer.ipynb)<br>🎓 [predict API](./鳶尾花隨機森林分類器/practice_predict_api_answer.ipynb)<br>🎓 [gradio UI](./鳶尾花隨機森林分類器/practice_gradio_ui_answer.ipynb) | 🌸 [專案講義](./鳶尾花隨機森林分類器/README.md) |
| **糖尿病罹病風險評估** | 智慧醫療 | 🩺 [邏輯迴歸 (二元分類與勝算比)](../邏輯迴歸/README.md)<br>📈 [評估指標 (混淆矩陣與ROC-AUC)](../評估指標/README.md) | 🩺 [03_diabetes_logistic_regression.ipynb](../邏輯迴歸/03_diabetes_logistic_regression.ipynb) | 🎓 [train API](./糖尿病風險評估-邏輯迴歸/practice_train_api_answer.ipynb)<br>🎓 [predict API](./糖尿病風險評估-邏輯迴歸/practice_predict_api_answer.ipynb)<br>🎓 [gradio UI](./糖尿病風險評估-邏輯迴歸/practice_gradio_ui_answer.ipynb) | 🩺 [專案講義](./糖尿病風險評估-邏輯迴歸/README.md) |
| **心臟衰竭生存風險預測** | 重症醫療 | 🌲 [決策樹集成模型 (隨機森林與預剪枝)](../決策樹集成模型/README.md)<br>📈 [評估指標 (特徵重要性與AUC)](../評估指標/README.md) | 🫀 [05_random_forest_heart_failure.ipynb](../決策樹集成模型/05_random_forest_heart_failure.ipynb) | 🎓 [train API](./心臟衰竭生存風險預測-隨機森林/practice_train_api_answer.ipynb)<br>🎓 [predict API](./心臟衰竭生存風險預測-隨機森林/practice_predict_api_answer.ipynb)<br>🎓 [gradio UI](./心臟衰竭生存風險預測-隨機森林/practice_gradio_ui_answer.ipynb) | 🫀 [專案講義](./心臟衰竭生存風險預測-隨機森林/README.md) |

---

## 🗺️ 學習路徑建議：進入模型部署前必備的演算法知識庫

在進行任何雲端微服務（FastAPI）或網頁介面（Gradio）的部署之前，建議學生**務必先掌握底層演算法的核心原理與評估指標**。以下是各專案對應的前置知識與關聯資料夾指引：

### 1. 💼 薪資預測 ➔ 先修：線性迴歸家族
* **先修演算法資料夾**：
  * 📐 [**簡單線性迴歸 (Simple Linear Regression)**](../簡單線性迴歸/README.md)：理解損失函數（MSE）、手動梯度下降法（Gradient Descent）與斜率/截距的幾何含意。
  * 📊 [**多元線性迴歸 (Multiple Linear Regression)**](../多元線性迴歸/README.md)：掌握多特徵矩陣運算、文字特徵編碼（`OrdinalEncoder` vs `OneHotEncoder`）、特徵縮放（`StandardScaler`）以及正則化抗過擬合（Ridge 與 Lasso）。
* **為什麼部署需要它**：部署端點 `/predict` 時若未打包 `OrdinalEncoder` 與 `StandardScaler`，將導致特徵尺度不一或學歷編碼順序倒轉（$R^2$ 驟降至 0.22）。

---

### 2. 🌸 鳶尾花品種分類 ➔ 先修：決策樹與集成森林
* **先修演算法資料夾**：
  * 🌿 [**樹狀模型 (Decision Trees)**](../樹狀模型/README.md)：理解資訊增益（Information Gain）、吉尼不純度（Gini Impurity）與單一決策樹容易過擬合的特性。
  * 🌲 [**決策樹集成模型 (Ensembles of Decision Trees)**](../決策樹集成模型/README.md)：掌握 Bagging（裝袋法）、Bootstrap 自助隨機抽樣、`max_features` 特徵隨機化，以及 100 棵樹的投票機制。
* **為什麼部署需要它**：線上熱重載端點 `/train` 允許使用者即時調整 `n_estimators` 與 `max_depth`，學生需具備樹木深度與模型複雜度平衡的概念。

---

### 3. 🩺 糖尿病罹病風險評估 ➔ 先修：邏輯迴歸與分類評估指標
* **先修演算法資料夾**：
  * 🩺 [**邏輯迴歸 (Logistic Regression)**](../邏輯迴歸/README.md)：理解線性組合如何透過 Sigmoid 函數擠壓至 $[0, 1]$ 成為機率、正則化強度倒數 $C$ 的意義、以及臨床勝算比（Odds Ratio, $e^\beta$）。
  * 📈 [**評估指標 (Evaluation Metrics)**](../評估指標/README.md)：理解醫療篩檢中為避免漏診所重視的召回率（Recall / Sensitivity）、混淆矩陣、以及 ROC-AUC 曲線。
* **為什麼部署需要它**：Gradio 介面即時繪製「危險因子勝算比水平長條圖」，若不理解 Odds Ratio 的統計意義，將無法向醫護人員解讀血糖或年齡的風險加權。

---

### 4. 🫀 心臟衰竭生存風險預測 ➔ 先修：隨機森林預剪枝與臨床特徵重要性
* **先修演算法資料夾**：
  * 🌲 [**決策樹集成模型 (Ensembles of Decision Trees)**](../決策樹集成模型/README.md)：學習臨床多變數分析中隨機森林的抗雜訊優勢、預修剪（Pre-pruning, `min_samples_split`）與特徵重要性（Feature Importances）計算。
  * 📈 [**評估指標 (Evaluation Metrics)**](../評估指標/README.md)：理解不平衡醫療資料集下的精確度與 AUC 評價。
* **為什麼部署需要它**：系統前端即時呈現 11 項生化指標的 Feature Importance 排名（如肌酸酐與射血分數），學生需理解這些權重來自隨機森林所有決策樹節點不純度總減少量的平均。
