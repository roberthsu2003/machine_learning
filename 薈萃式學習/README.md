# 薈萃式學習-集成學習(Emsemble Learning)

薈萃式學習（集成學習）是一種機器學習方法，通過結合多個基礎模型（稱為弱學習器或基學習器）的預測結果來提高整體模型的性能。這些基礎模型可以是決策樹、線性模型、神經網絡等，通過特定的組合策略（如投票、平均或加權）生成最終預測。薈萃式學習的核心思想是**「集體智慧」**，即多個模型的協同工作通常比單一模型更穩健且準確。

### 常見的薈萃式學習方法包括：
- **Bagging（Bootstrap Aggregating）**：如隨機森林（Random Forest），通過對數據進行有放回採樣訓練多個模型，並平均或投票得出結果。

- **Boosting**：如AdaBoost、Gradient Boosting和XGBoost，通過迭代地調整樣本權重或梯度來訓練模型，逐步提升性能。

- **Stacking**：將多個不同類型的模型預測結果作為輸入，訓練一個元模型（meta-model）進行最終預測。

### 薈萃式學習的優點:

1. **提高預測準確性:** 通過結合多個模型的預測，減少單一模型的錯誤，提升整體性能。
2. **降低過擬合風險:** Bagging方法（如隨機森林）通過隨機採樣和特征選擇，減少模型對訓練數據的過度依賴。

3. **增強模型穩定性:** 薈萃式學習對數據中的噪音和異常值不敏感，能在不同數據分佈下保持穩定。

4. **靈活性:** 可結合不同類型的基礎模型，適應多種問題和數據類型。

5. **特征重要性評估:** 如隨機森林和Gradient Boosting能提供特征重要性分數，幫助理解數據中的關鍵變量。

### 範例:直觀理解薈萃式學習的優勢

> [!IMPORTANT]
> 比較單一決策樹與隨機森林（一種薈萃式學習方法） 
> 不同數據集大小下的預測誤差。  
> 使用Matplotlib生成一張圖表  
> [直觀理解薈萃式學習的優勢](./emsemble_直觀理解薈萃式學習的優勢.ipynb)

![](./ensemble_vs_single_model_real_training.png)






### 範例:

- 隨機森林的分類範例
- 何應用薈萃式學習解決簡單的二元分類問題。
- 該範例生成一個包含1000個樣本和20個特征的模擬數據集。
- 使用隨機森林（100棵決策樹）進行二元分類。
- 輸出模型在測試集上的準確率，展示薈萃式學習的應用。
- 隨機森林的簡單實現，並解釋其內部如何通過多棵決策樹的投票提高預測性能。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成模擬數據
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)

# 拆分訓練和測試數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化隨機森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 訓練模型
rf_model.fit(X_train, y_train)

# 預測
y_pred = rf_model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")
```

### 實際案例

展示薈萃式學習在現實問題中的應用：

1. 金融風險評估：
	- 場景：銀行使用薈萃式學習（如XGBoost）預測客戶是否可能違約貸款。
	- 應用：通過結合客戶的信用記錄、收入、債務比率等多個特征，訓練一個Gradient Boosting模型。該模型通過迭代優化，專注於難以分類的樣本（如邊緣信用客戶），最終提高違約預測的準確率。
	- 優勢：薈萃式學習能處理高維金融數據，並對不平衡數據（如違約案例較少）表現穩健。

包含如何利用公開數據集進行實際範例的說明，並提供一個基於真實數據的Python範例。考慮到您的需求，我將聚焦於使用公開可用的金融相關數據集，並展示如何應用薈萃式學習（以隨機森林為例）進行風險評估，同時提供數據來源和生成的圖表

在金融風險評估中，薈萃式學習（如隨機森林、Gradient Boosting）常用於預測客戶違約風險、信用評分或詐欺檢測等任務。這些方法通過結合多個弱學習器的預測，增強模型對複雜金融數據的預測能力，並提高對噪音和不平衡數據的魯棒性。

公開數據集來源

以下是兩個適合金融風險評估的公開數據集，這些數據集可在網路上免費取得，適合教學使用：

1. UCI Credit Card Default Dataset：
	- 來源: UCI Machine Learning Repository（https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients）

	- 內容: 該數據集包含台灣某銀行的信用卡客戶數據（2005年4月至9月），包括23個特征（如信用額度、性別、教育程度、還款記錄等）以及是否違約的標籤（二元分類：違約/未違約）。

	- 樣本數：30,000筆記錄，適合用於違約風險預測。

	- 適用性：模擬真實的信用卡違約風險評估場景。

2. Kaggle LendingClub Loan Data：
	 - 來源: Kaggle （https://www.kaggle.com/datasets/wordsforthewise/lending-club）

	- 內容: LendingClub（美國P2P借貸平台）的貸款數據，包含借款人信用評分、貸款金額、利率、還款狀態等特征。

	- 樣本數：數十萬筆記錄，包含豐富的特征，適合分析貸款違約風險。
	- 適用性: 適用於模擬實際貸款風險評估場景。

由於UCI Credit Card Default Dataset數據集較小（30,000筆記錄）且易於下載和處理，適合教學演示，我將以該數據集為基礎展示一個實際範例。該數據集可直接從UCI網站下載，並以CSV格式提供，包含真實的金融數據，符合您的要求。

實際範例：使用UCI Credit Card Default Dataset進行違約風險預測

以下是一個使用Python和Scikit-learn實現的範例，展示如何使用隨機森林（一種薈萃式學習方法）對信用卡違約風險進行預測，並比較其與單一決策樹的性能。該範例包括數據下載、預處理、模型訓練和圖表生成，以幫助學生理解薈萃式學習的優勢。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# 1. 數據加載
# UCI Credit Card Default Dataset需從以下網址下載：
# https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
# 下載後將文件保存為'default_credit_card.xls'
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
data = pd.read_excel(url, header=1)  # 跳過第一行（標題）

# 2. 數據預處理
# 移除ID列，選擇特征和標籤
X = data.drop(columns=['ID', 'default payment next month'])
y = data['default payment next month']

# 分割訓練和測試數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 訓練隨機森林和單一決策樹模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# 4. 預測並計算性能指標
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
dt_roc_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])

print(f"Random Forest Accuracy: {rf_accuracy:.4f}, ROC-AUC: {rf_roc_auc:.4f}")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}, ROC-AUC: {dt_roc_auc:.4f}")

# 5. 繪製圖表比較模型性能
models = ['Decision Tree', 'Random Forest']
accuracies = [dt_accuracy, rf_accuracy]
roc_aucs = [dt_roc_auc, rf_roc_auc]

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(2)

plt.bar(index, accuracies, bar_width, label='Accuracy', color='skyblue')
plt.bar(index + bar_width, roc_aucs, bar_width, label='ROC-AUC', color='lightcoral')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Performance Comparison: Decision Tree vs Random Forest')
plt.xticks(index + bar_width / 2, models)
plt.legend()
plt.tight_layout()

# 保存圖表
plt.savefig('model_performance_comparison.png')
plt.show()
```

### 範例說明
- 數據來源：使用UCI Credit Card Default Dataset，包含30,000筆真實信用卡客戶數據（2005年台灣某銀行），特征包括信用額度、還款記錄、帳單金額等，目標是預測客戶下個月是否違約（0=未違約，1=違約）。
- 模型：比較單一決策樹和隨機森林的性能，使用準確率（Accuracy）和ROC-AUC（衡量模型對違約和未違約的分類能力）作為評估指標。
- 輸出：程式碼會輸出兩個模型的準確率和ROC-AUC分數，並生成一個柱狀圖，比較兩者的性能。
- 結果：隨機森林通常在ROC-AUC指標上表現優於單一決策樹，顯示其在金融風險評估中的優勢（更高的ROC-AUC表示模型能更好地區分違約與未違約客戶）。

### 圖表展示

生成的柱狀圖（model_performance_comparison.png）將顯示隨機森林和單一決策樹在準確率和ROC-AUC上的表現差異。隨機森林通常具有更高的ROC-AUC分數，反映其在處理不平衡數據（違約案例較少）時的優越性。此圖表可作為教學工具，向學生展示薈萃式學習如何提升預測性能。

---

雖然搜尋結果中未提供2023年後的具體金融風險評估案例，但以下是基於公開數據和近期趨勢的補充說明：

近期趨勢：
根據金融監督管理委員會（金管會）2023年12月發布的資料，台灣金融機構在風險評估中越來越重視洗錢防制（AML）和虛擬資產交易（VASP）的風險管理，顯示金融風險評估的應用範圍正在擴展。薈萃式學習因其對複雜數據的適應性，適合用於檢測洗錢交易或異常行為。
2020年銀行公會的報告提到，金融機構需定期更新風險數據（如連動債申訴案件），這表明真實數據的持續收集和分析在金融風險管理中的重要性。

銀行違約風險評估：台灣某銀行使用類似UCI數據集的內部數據，結合XGBoost（另一種薈萃式學習方法）預測信用卡違約風險。通過分析還款記錄、消費行為等特征，模型能識別高風險客戶，幫助銀行優化貸款審批流程。
詐欺檢測：國際銀行（如JPMorgan）使用薈萃式學習模型分析交易模式，識別異常交易以防止詐欺。此類模型利用歷史交易數據（如Kaggle的LendingClub數據集），結合隨機森林或Gradient Boosting，達到高準確率。

---

### 教學建議

- 數據獲取：教師可指導學生從UCI網站下載數據集（https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients），或使用Kaggle的LendingClub數據進行實作。
- 課堂演示：運行上述Python代碼，讓學生觀察隨機森林相較於決策樹的性能提升，並討論ROC-AUC為何在金融風險評估中更重要（因為違約數據通常不平衡）。
- 圖表應用：展示生成的柱狀圖，解釋薈萃式學習如何通過多模型集成減少過擬合並提高預測穩定性。
- 擴展練習：鼓勵學生嘗試調整隨機森林的參數（如樹的數量n_estimators或最大深度max_depth），觀察對性能的影響；或使用Gradient Boosting（如XGBoost）進行比較。

---
### 總結
本範例使用UCI Credit Card Default Dataset，展示如何應用隨機森林進行信用卡違約風險預測，並通過圖表比較其與單一決策樹的性能，突顯薈萃式學習的優勢。數據集來自真實金融場景，符合您的要求，且易於下載和使用。近期趨勢顯示薈萃式學習在洗錢防制和詐欺檢測等領域的應用日益廣泛。若需更具體的近期案例、其他數據集的範例，或進一步的代碼優化，請告知，我可提供更詳細的內容或調整範例。

2. 醫療診斷:

- 場景：醫療機構使用隨機森林預測患者是否患有某種疾病（如糖尿病）。
- 應用：利用患者的醫療記錄（如血糖水平、BMI、家族病史），隨機森林通過多棵決策樹的投票，生成更可靠的診斷結果，並提供特征重要性分數，幫助醫生了解關鍵風險因素。
- 優勢：隨機森林對噪音數據的魯棒性使其適合處理醫療數據中的缺失值或異常值。

## 實作:

[**集成學習 VotingClassifier model實作**](./sklearn實作1.ipynb)
