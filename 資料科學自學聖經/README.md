# 🎯 《資料科學自學聖經》數據集與機器學習模型導向速查地圖

歡迎來到 **《資料科學自學聖經》數據集與學習資源庫**！
本資源庫已將所有隨書數據集與 Jupyter Notebook 範例按照 **「機器學習任務與適用模型」** 進行系統化重新目錄分級。

無論你是想找**特定數據集**來練習，還是想針對**特定演算法（如隨機森林、SVM、K-Means 等）**尋找適合的數據，都能透過本 README 快速定位！

---

## 🗺️ 目錄結構一覽 (Folder Overview)

```text
資料科學自學聖經/
├── 📁 01_數據預處理與基礎分析/   # Pandas、Numpy、Matplotlib 資料清洗與 EDA 基礎練習
├── 📁 02_迴歸模型數據集/        # 簡單/多元線性迴歸、Ridge/Lasso 數值預測
├── 📁 03_分類模型數據集/        # 邏輯迴歸、決策樹、隨機森林、SVM、KNN 等二分類與多分類
├── 📁 04_集群與關聯規則/        # K-Means 客戶分群與 Apriori 超市購物籃分析
├── 📁 05_自然語言與文本分類/    # 新聞標題文字分類 (TF-IDF + Naive Bayes/Logistic Regression)
├── 📁 06_時間序列與RNN/         # 台股股價歷史數據與 RNN/LSTM 時間序列預測
├── 📁 07_電腦視覺與深度學習/    # MNIST 手寫數字 (DNN)、貓狗二分類 (CNN)、花朵遷移學習與 YOLOv4
└── 📁 08_自動化推播與應用/      # 模型成果整合與 LINE Notify 自動化推播
```

---

## 📊 一、數據集 ➡️ 推薦機器學習模型對照表 (Master Dataset Matrix)

| 數據集名稱 | 檔案相對路徑 | 應用情境 / 領域 | 問題類型 | 適合採用的模型 / 演算法 | 推薦評估指標 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **廣告銷售額** | [`02_迴歸模型數據集/advSale.csv`](./02_迴歸模型數據集/advSale.csv) | 電視/廣播/社群廣告預測銷售量 | 連續迴歸 (Regression) | **簡單線性迴歸**、**多元線性迴歸** | $R^2$, MSE, MAE |
| **地區房價數據** | [`02_迴歸模型數據集/housePrice.csv`](./02_迴歸模型數據集/housePrice.csv) | 犯罪率、捷運距離預測房價 | 連續迴歸 (Regression) | **多元線性迴歸**、**嶺迴歸 (Ridge)**、**Lasso** | $R^2$, RMSE |
| **波士頓房價** | [`02_迴歸模型數據集/BostonHousing.csv`](./02_迴歸模型數據集/BostonHousing.csv) | 經典波士頓房價特徵預測 | 連續迴歸 (Regression) | **多元線性迴歸**、**決策樹迴歸** | $R^2$, MSE |
| **鐵達尼號生存** | [`03_分類模型數據集/titanic.csv`](./03_分類模型數據集/titanic.csv) | 乘客特徵預測船難生存率 | 二元分類 (Binary Class) | **邏輯迴歸**、**決策樹**、**隨機森林**、**XGBoost** | Accuracy, ROC-AUC, F1 |
| **乳癌腫瘤診斷** | [`03_分類模型數據集/breastCancer.csv`](./03_分類模型數據集/breastCancer.csv) | 細胞厚度與形狀判讀良/惡性 | 二元分類 (Binary Class) | **邏輯迴歸**、**SVM (支援向量機)**、**k-NN** | Recall, Precision, F1 |
| **紅酒品種分類** | [`03_分類模型數據集/wine.csv`](./03_分類模型數據集/wine.csv) | 酒精與黃酮類特徵區分 3 種紅酒 | 多類別分類 (Multi-Class) | **k-NN 分類器**、**高斯貝氏 (GaussianNB)**、**SVM** | Accuracy, Confusion Matrix |
| **稻米品種分類** | [`03_分類模型數據集/rice.csv`](./03_分類模型數據集/rice.csv) | 稻米長寬與面積辨識品種 | 二元/多類別分類 | **決策樹 (Decision Tree)**、**隨機森林** | Accuracy, F1-Score |
| **羽球天氣決策** | [`03_分類模型數據集/raw_data.csv`](./03_分類模型數據集/raw_data.csv) | 風速狀況預測能否打羽球 | 二元分類 (Binary Class) | **樸素貝氏 (Naive Bayes)**、**單一決策樹** | Accuracy |
| **銀行客戶訂購** | [`03_分類模型數據集/客戶聯絡狀況資料檔.csv`](./03_分類模型數據集/客戶聯絡狀況資料檔.csv) | 客戶年齡與聯絡次數預測定存 | 二元分類 (含類別編碼) | **邏輯迴歸**、**隨機森林**、**梯度提升樹** | F1-Score, ROC-AUC |
| **客戶消費分群** | [`04_集群與關聯規則/customer.csv`](./04_集群與關聯規則/customer.csv) | 年齡、收入與消費指數做客戶畫像 | 非監督集群 (Clustering) | **K-Means 集群**、**階層式集群 (Hierarchical)** | Silhouette Score, Elbow Method |
| **超市購物籃分析** | [`04_集群與關聯規則/orders.csv.zip`](./04_集群與關聯規則/orders.csv.zip) | 分析哪些商品常被一起購買 | 關聯規則 (Association) | **Apriori 演算法**、**FP-Growth** | Support, Confidence, Lift |
| **新聞標題分類** | [`05_自然語言與文本分類/toutiao_cat_data.txt.zip`](./05_自然語言與文本分類/toutiao_cat_data.txt.zip) | 今日頭條新聞文本類別識別 | 文本多分類 (NLP) | **TF-IDF + Naive Bayes**、**TF-IDF + 邏輯迴歸** | Accuracy, Macro-F1 |
| **台股歷史股價** | [`06_時間序列與RNN/twstock_all.csv`](./06_時間序列與RNN/twstock_all.csv) | 2015-2021 台股開盤/收盤價 | 時間序列 (Time Series) | **RNN**、**LSTM**、**GRU** | RMSE, MAPE |
| **MNIST 手寫數字** | [`07_電腦視覺與深度學習/mnist500.zip`](./07_電腦視覺與深度學習/mnist500.zip) | 28x28 手寫數字圖像識別 (0-9) | 圖像多分類 (Vision) | **多層感知機 (MLP)**、**CNN (卷積神經網路)** | Accuracy |
| **貓狗影像二分類** | [`07_電腦視覺與深度學習/Cat1.jpg`](./07_電腦視覺與深度學習/Cat1.jpg) 等 | 貓咪與狗狗照片辨識 | 圖像二分類 (Vision) | **CNN (卷積神經網路)** | Accuracy, Confusion Matrix |
| **花朵品種辨識** | [`07_電腦視覺與深度學習/flower.zip`](./07_電腦視覺與深度學習/flower.zip) | 雛菊/玫瑰/向日葵等多種花朵 | 圖像多分類 (Vision) | **遷移學習 (Transfer Learning: ResNet/VGG)** | Accuracy, Top-5 Accuracy |

---

## 🤖 二、按機器學習演算法檢索數據集 (Model-First Lookup Index)

如果你正在學習特定的演算法，請參照下表尋找最合適的練習數據集與範例 Notebook：

### 1. 迴歸演算法 (Regression Algorithms)
* **簡單 / 多元線性迴歸 (Simple / Multiple Linear Regression)**
  * 推薦數據集：[`02_迴歸模型數據集/advSale.csv`](./02_迴歸模型數據集/advSale.csv)（廣告金額預測銷售額）
  * 範例 Notebook：[`02_迴歸模型數據集/Ch09_機器學習：監督式學習迴歸演算法.ipynb`](./02_迴歸模型數據集/Ch09_機器學習：監督式學習迴歸演算法.ipynb)
* **嶺迴歸與 Lasso (Ridge & Lasso Regression)**
  * 推薦數據集：[`02_迴歸模型數據集/housePrice.csv`](./02_迴歸模型數據集/housePrice.csv)（多變數房價預測）

### 2. 分類演算法 (Classification Algorithms)
* **邏輯迴歸 (Logistic Regression)**
  * 推薦數據集：[`03_分類模型數據集/titanic.csv`](./03_分類模型數據集/titanic.csv)（鐵達尼號生存預測）
  * 觀念 Notebook：[`03_分類模型數據集/SigmoidFunction.ipynb`](./03_分類模型數據集/SigmoidFunction.ipynb)
* **k-近鄰演算法 (k-Nearest Neighbors, k-NN)**
  * 推薦數據集：[`03_分類模型數據集/wine.csv`](./03_分類模型數據集/wine.csv)（紅酒成分分類）
* **樸素貝氏分類器 (Naive Bayes Classifier)**
  * 推薦數據集：[`03_分類模型數據集/raw_data.csv`](./03_分類模型數據集/raw_data.csv)（天氣決策）、[`05_自然語言與文本分類/toutiao_cat_data.txt.zip`](./05_自然語言與文本分類/toutiao_cat_data.txt.zip)（文本分類）
* **決策樹與隨機森林 (Decision Trees & Random Forest)**
  * 推薦數據集：[`03_分類模型數據集/rice.csv`](./03_分類模型數據集/rice.csv)（稻米品種）、[`03_分類模型數據集/titanic.csv`](./03_分類模型數據集/titanic.csv)
  * 範例 Notebook：[`03_分類模型數據集/Iris_DecisionTree.ipynb`](./03_分類模型數據集/Iris_DecisionTree.ipynb)
* **支援向量機 (Support Vector Machine, SVM)**
  * 推薦數據集：[`03_分類模型數據集/breastCancer.csv`](./03_分類模型數據集/breastCancer.csv)（乳癌腫瘤診斷）

### 3. 非監督式學習 (Unsupervised Learning)
* **K-Means 集群分析 (K-Means Clustering)**
  * 推薦數據集：[`04_集群與關聯規則/customer.csv`](./04_集群與關聯規則/customer.csv)（客戶消費特徵分群）
  * 範例 Notebook：[`04_集群與關聯規則/Ch07_機器學習：非監督式學習.ipynb`](./04_集群與關聯規則/Ch07_機器學習：非監督式學習.ipynb)
* **購物籃關聯分析 (Apriori Association Rules)**
  * 推薦數據集：[`04_集群與關聯規則/orders.csv.zip`](./04_集群與關聯規則/orders.csv.zip) & [`products.csv`](./04_集群與關聯規則/products.csv)

### 4. 深度學習與先進領域 (Deep Learning & Advanced Topics)
* **時間序列循環神經網路 (RNN / LSTM)**
  * 推薦數據集：[`06_時間序列與RNN/twstock_all.csv`](./06_時間序列與RNN/twstock_all.csv)（台股歷史股價）
  * 範例 Notebook：[`06_時間序列與RNN/深度學習：循環神經網路(RNN).ipynb`](./06_時間序列與RNN/深度學習：循環神經網路(RNN).ipynb)
* **圖像卷積神經網路 (CNN)**
  * 推薦數據集：[`07_電腦視覺與深度學習/Cat1.jpg`](./07_電腦視覺與深度學習/Cat1.jpg) 等（貓狗圖片）
  * 範例 Notebook：[`07_電腦視覺與深度學習/深度學習：卷積神經網路(CNN).ipynb`](./07_電腦視覺與深度學習/深度學習：卷積神經網路(CNN).ipynb)
* **遷移學習 (Transfer Learning)**
  * 推薦數據集：[`07_電腦視覺與深度學習/flower.zip`](./07_電腦視覺與深度學習/flower.zip)（花朵圖像）
  * 範例 Notebook：[`07_電腦視覺與深度學習/預訓練模型及遷移學習.ipynb`](./07_電腦視覺與深度學習/預訓練模型及遷移學習.ipynb)

---

## 🚀 三、學生快速上手程式碼 (Quick-Start Examples)

### 範例 1：多元線性迴歸 (預測廣告效益)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. 載入數據集
df = pd.read_csv("02_迴歸模型數據集/advSale.csv")

# 2. 定義特徵 (X) 與標籤 (y)
X = df[['電視廣告', '廣播廣告', '社群媒體廣告']]
y = df['銷售額']

# 3. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 評估模型
y_pred = model.predict(X_test)
print(f"決定係數 (R^2 Score): {r2_score(y_test, y_pred):.4f}")
print(f"均方誤差 (MSE): {mean_squared_error(y_test, y_pred):.4f}")
```

---

### 範例 2：隨機森林分類 (鐵達尼號生存預測)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 載入數據集
df = pd.read_csv("03_分類模型數據集/titanic.csv")

# 2. 簡單資料預處理 (填補年齡缺失值、編碼性別)
df['age'] = df['age'].fillna(df['age'].median())
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# 3. 選取特徵與標籤
features = ['pclass', 'sex', 'age']
X = df[features]
y = df['survived']

# 4. 訓練集切分與模型擬合
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. 評估分類結果
y_pred = clf.predict(X_test)
print(f"分類準確度 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

### 範例 3：K-Means 集群分析 (客戶消費特徵分群)

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. 載入數據集
df = pd.read_csv("04_集群與關聯規則/customer.csv")

# 2. 選取分群特徵並進行特徵標準化
X = df[['年齡', '收入(千)', '消費指數(1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 建立 K-Means 模型 (設定 5 個群集)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. 計算輪廓係數 (Silhouette Score)
score = silhouette_score(X_scaled, df['Cluster'])
print(f"K-Means 分群輪廓係數: {score:.4f}")
print(df.groupby('Cluster').mean())
```

---

祝你在機器學習與數據科學的學習旅程中收穫滿滿！🎉
