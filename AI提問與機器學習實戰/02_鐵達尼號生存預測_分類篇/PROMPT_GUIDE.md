# 💬 範例 02：向 AI 提問的五階段實戰手冊 (分類篇)

本手冊教導初學者如何面對「有缺失值、有類別不平衡、甚至隱藏作弊特徵」的真實世界分類數據。請依照以下五個階段，逐步將提示詞複製給你的 AI 工具。

---

## 📌 提問心法：分類問題的致命陷阱

1. ❌ **陷阱一：把未來發生的事當成特徵（資料洩漏 Data Leakage）**
   - 資料中有一個欄位叫 `boat`（救生艇號碼）。初學者直接把所有欄位塞給 AI，AI 做出高達 99% 準確率，但這在實務上完全沒有意義（船剛撞冰山時根本不知道誰會上哪艘救生艇）。
2. ❌ **陷阱二：被準確率 (Accuracy) 欺騙**
   - 資料中只有 34% 人生還，如果模型「什麼都不學，全部預測死亡」，準確率居然也有 66%！因此必須逼 AI 解釋 **Recall（召回率）** 與 **Precision（精確率）**。
3. ❌ **陷阱三：隨機切分導致比例偏斜**
   - 分類問題切分 Train/Test 時，必須提醒 AI 加上 `stratify=y`。

---

## 🚀 階段 0：角色設定與資料餵入

### 步驟 0-1：在 Python 中取得資料摘要
在 Jupyter Notebook 中執行：
```python
import pandas as pd
df = pd.read_csv("data/titanic.csv")
print("=== 資料筆數與缺失值統計 ===")
print(df.isnull().sum())
print("\n=== 前 3 筆樣本 ===")
print(df.head(3))
```

### 步驟 0-2：複製以下 Prompt 給 AI 🚀
> [!TIP]
> **📋 複製這段提示詞給 AI：**
> 
> ```markdown
> 你現在是一位專業的「資料科學導師」。我是一位機器學習初學者，手邊有一份經典的鐵達尼號乘客數據集 `data/titanic.csv`。
> 
> 以下是資料集摘要：
> 1. 總筆數：1,313 筆，目標變數是 `survived`（1=生存，0=遇難，生存者約佔 34%）。
> 2. 欄位清單與缺失狀況：
>    - `row.names` (int): 乘客序號 (無缺失)
>    - `pclass` (object): 艙等 1st, 2nd, 3rd (無缺失)
>    - `name` (object): 乘客姓名 (無缺失)
>    - `age` (float): 年齡 (有 680 筆缺失值)
>    - `embarked` (object): 登船港口 (有 492 筆缺失值)
>    - `home.dest` (object): 目的地 (有 559 筆缺失值)
>    - `room`, `ticket` (object): 房號與票號 (缺失超過 90%)
>    - `boat` (object): 救生艇號碼 (有 966 筆缺失值)
>    - `sex` (object): 性別 male / female (無缺失)
> 
> 我的目標：根據乘客在「開船前」的基本特徵，預測其在船難中的生還機率 (`survived`)。
> 
> 【互動規範】：
> 1. 請全程使用台灣繁體中文專業術語回答我。
> 2. 請先幫我進行「特徵審查」：以常理與機器學習規範，哪些欄位應該剔除？請特別說明 `boat` 這個欄位適不適合當作特徵？為什麼？
> 3. 請不要直接產出完整程式碼，請列出後續五個步驟，等我確認後再進行下一步。
> ```

---

## 🔍 階段 1：特徵挑選與資料探索 (EDA)

當 AI 指出 `boat` 屬於「未來作弊欄位（Data Leakage）」並建議剔除無效欄位後，進行可視化探索。

> [!TIP]
> **📋 複製這段提示詞給 AI：**
> 
> ```markdown
> 我們決定只保留具有合理預測價值的特徵：`pclass`（艙等）、`sex`（性別）、`age`（年齡）、`embarked`（登船港口），目標是 `survived`。
> 
> 請提供一段 Python 探索與可視化程式碼：
> 1. 繪製一張長條圖，比較「男性 vs 女性」的生還率差異。
> 2. 繪製一張長條圖，比較「不同艙等 (1st, 2nd, 3rd)」的生還率差異。
> 3. 請確保圖表中文字型正確顯示（支援 `mplfonts` 或無方塊亂碼）。
> 4. 執行完後，請用通俗語言解釋：「女性」與「頭等艙」乘客的生還機率是否真的顯著偏高？這反映出什麼現實背景？
> ```

---

## 🛠️ 階段 2：缺失值填補與 Pipeline 建構

> [!TIP]
> **📋 複製這段提示詞給 AI：**
> 
> ```markdown
> 現在我們要對特徵進行前處理。我注意到：
> - `age` 是連續數值，有許多空值。
> - `pclass`, `sex`, `embarked` 是類別型欄位，且 `embarked` 也有空值。
> 
> 請教導我並寫出 Scikit-Learn 處理程式碼：
> 1. 數值特徵 (`age`)：請使用 `SimpleImputer(strategy='median')` 填補中位數，並搭配 `StandardScaler` 標準化。為什麼年齡補值建議用中位數而非平均數？
> 2. 類別特徵：請用 `SimpleImputer(strategy='most_frequent')` 補眾數，並用 `OneHotEncoder(handle_unknown='ignore')` 編碼。
> 3. 使用 `ColumnTransformer` 將兩者封裝。
> 4. 切分資料集：80% 訓練集、20% 測試集。請務必加上 `stratify=y` 與固定 `random_state=42`。請向我解釋為什麼分類問題必須加上 `stratify=y`？
> ```

---

## 🤖 階段 3：分類模型選型與訓練

> [!TIP]
> **📋 複製這段提示詞給 AI：**
> 
> ```markdown
> 請幫我建立兩個二元分類模型進行評估：
> 1. 基準模型 (Baseline)：邏輯迴歸 (LogisticRegression)
> 2. 樹狀集成模型：隨機森林分類器 (RandomForestClassifier, random_state=42)
> 
> 請滿足以下規範：
> - 透過 `sklearn.pipeline.Pipeline` 結合剛才的前處理器與模型。
> - 在訓練集上使用 5 折分層交叉驗證 (StratifiedKFold)，以 `f1` 作為打分指標。
> - 請先不要看測試集，先列出這兩個模型在訓練集上的交叉驗證 F1-score 平均值與標準差。
> ```

---

## 📊 階段 4：測試集評估與混淆矩陣指標解讀

> [!TIP]
> **📋 複製這段提示詞給 AI：**
> 
> ```markdown
> 現在我們要拿封存的測試集 (X_test, y_test) 來評估這兩個模型。
> 
> 請提供評估程式碼，包含：
> 1. 印出兩個模型的「分類報告 (classification_report)」，列出 Precision, Recall, F1-score 與 Accuracy。
> 2. 使用 Seaborn 繪製「混淆矩陣熱力圖 (Confusion Matrix)」，標示出 TP (真正例)、FP (假正例)、TN (真負例)、FN (假負例)。
> 
> 【白話解說需求】：
> 我是新手，看不懂專業術語，請用生活化比喻回答：
> - 什麼是「精確率 (Precision)」？什麼是「召回率 (Recall)」？
> - 在鐵達尼號救援的情境中，「把活人誤判為死亡」與「把死者誤判為生還」，哪一個對應到 Precision？哪一個對應到 Recall？
> - 最終你推薦我們採用哪一個模型？
> ```

---

## 🔮 階段 5：自訂虛擬乘客推論試算 (Inference)

> [!TIP]
> **📋 複製這段提示詞給 AI：**
> 
> ```markdown
> 請寫出一個 Python 預測函式 `predict_passenger_survival(pipeline, passenger_dict)`：
> 1. 輸入乘客資訊字典，輸出預測結果（生還 / 遇難）以及「生還機率百分比 (predict_proba)」。
> 2. 請幫我測試以下 3 位虛擬乘客的生還機率：
>    - 乘客 A（電影女主角 Rose）：頭等艙 (`1st`)、女性 (`female`)、年齡 17 歲、從 Southampton 登船。
>    - 乘客 B（電影男主角 Jack）：三等艙 (`3rd`)、男性 (`male`)、年齡 20 歲、從 Southampton 登船。
>    - 乘客 C（三等艙小男孩）：三等艙 (`3rd`)、男性 (`male`)、年齡 5 歲、從 Cherbourg 登船。
> 3. 請輸出三位乘客的預測生還機率，並分析「性別與艙等」對生存機率的巨大影響。
> ```

---

## 🛡️ 初學者防雷檢查清單 (AI Code Review Checklist)

- [ ] **是否成功排除 `boat`、`ticket`、`room` 等無效或作弊欄位？**
- [ ] **是否有對目標標籤使用 `stratify=y` 進行分層切分？**
- [ ] **缺失值補齊（Imputer）是否正確封裝在 Pipeline 內，且只在訓練集 fit？**
- [ ] **評估指標是否有看 Recall 與 F1-score，而不是盲目相信 Accuracy？**
