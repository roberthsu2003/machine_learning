# 範例 13：企業合約法律風險預警與審查分流系統

> **適用族群**：企業內部法務人員、法規遵循人員、中小企業負責人、合約管理專員  
> **使用工具**：VS Code + GitHub Copilot + uv + Python (Jupyter Notebook / Tkinter)  
> **預測類型**：多類別分類（Multiclass Classification: 低風險 / 中風險 / 高風險）

---

## 📖 業務情境與核心痛點

### 法務每天的痛：合約堆積如山，哪一份要先看？
在許多公司中，法務部門往往只有 1~2 人，卻要負責全公司業務部、採購部、IT 委外與經銷商的所有合約審查：
- 業務部急著做業績，催著法務「這份經銷合約今天能不能蓋章用印？」
- 外部廠商傳來的合約動輒 30~50 頁，裡面夾雜各種不對等的霸王條款（如：無損害賠償上限、由對方國外法院管轄、智財權全數讓與對方）。
- 法務若逐字細審，每份要花 2~3 小時，業務就會抱怨法務是「商業絆腳石」；但若草率放行，一旦發生違約爭議，公司可能面臨數百萬甚至上千萬的法律賠償責任。

### 🎯 機器學習可以幫法務預測什麼？
如果我們把過去審核過的 1,000 份歷史合約（包含合約金額、付款天數、重要條款勾選、合作廠商違約紀錄），以及法務主管最終裁定的「風險等級」整理成資料表：
1. **合約風險等級預測**：只要輸入新合約的商務要件與關鍵條款，機器學習能在 **0.1 秒內預測該合約為「低風險」、「中風險」還是「高風險」**。
2. **自動分流與時間管理**：
   - **低風險合約（約 35%）**：條件標準，可走「綠色免審通道」或授權單位主管直接簽署。
   - **中風險合約（約 40%）**：提供標準條款修改範本，由法務專員快速核對。
   - **高風險合約（約 25%）**：立刻發出紅字警告，由資深法務長集中火力逐條修改談判。
3. **客製化桌面 App**：法務人員桌面上隨時點開小視窗，勾選條款就能完成評估，並產生建議修改意見！

---

## 🛠️ 開發環境快速指引（使用 `uv`）

在開始實作前，請在 VS Code 終端機中進入本目錄，使用快速、現代的 Python 套件管理工具 `uv`：

```bash
# 1. 進入本範例目錄
cd 13_legal_contract_risk

# 2. 安裝本例所需之相依套件（Tkinter 為 Python 內建）
uv add pandas numpy scikit-learn matplotlib joblib ipykernel mplfonts
```

---

## 🚀 實作二階段流程指引

### 階段 A：`lesson.ipynb`（AI Copilot 陪跑探索與建模）

請在 VS Code 中開啟 `lesson.ipynb`，對 Copilot 下達以下提示詞逐步完成：

- **`A1 讀取與觀察資料`**：
  > 「請幫我讀取 `data/contract_risk.csv`，顯示前 5 筆資料、資料筆數、資料型態，並檢查是否有任何缺失值。接著統計目標標籤 `risk_level` 的分佈比例，並使用繁體中文繪製各合約類型的平均標的金額長條圖。」
- **`A2 特徵與資料切分`**：
  > 「請將 `contract_id` 剔除，以 `risk_level` 為目標標籤 y，其餘欄位為特徵 X。依據分層抽樣（stratify=y）以 8:2 切分訓練集與測試集，並固定隨機種子 `random_state=42`。請說明為什麼這裡必須使用分層抽樣？」
- **`A3 建立 Pipeline 與模型訓練`**：
  > 「請使用 `scikit-learn` 的 `ColumnTransformer` 與 `Pipeline`：對類別特徵 `contract_type` 做 One-Hot Encoding，數值特徵做 `StandardScaler`。分類演算法選用 `RandomForestClassifier`，在訓練集上進行 5 折交叉驗證（Cross Validation），並輸出訓練集的平均準確率。」
- **`A4 測試集驗收與混淆矩陣解讀`**：
  > 「請使用測試集（Test Set）進行最終預測，輸出分類報告（Classification Report：Precision、Recall、F1-Score），並以繁體中文繪製混淆矩陣熱力圖（Confusion Matrix）。最後請提取隨機森林的特徵重要性（Feature Importance），畫出對合約風險影響最大的前 5 項特徵。」
- **`A5 保存模型與元資料`**：
  > 「請將訓練完成的完整 Pipeline 模型保存至 `artifacts/model.joblib`，並將特徵名稱、類別清單、模型評估指標等資訊寫入 `artifacts/metadata.json`。」

---

### 階段 B：`app.py`（打造法務專屬 Tkinter 桌面預測 App）

完成模型訓練後，我們進入 `app.py`，利用 Python 內建的 Tkinter 打造一個免連網、隨開隨用的桌面視窗：

- **`B1 建立 Tkinter 介面`**：
  > 「請在 `app.py` 中使用 Python 內建的 `tkinter` 與 `ttk` 建立一個現代風格的『合約法律風險評估系統』桌面視窗。程式啟動時載入 `artifacts/model.joblib` 與 `metadata.json`。介面需包含：合約類型下拉選單、合約金額輸入框、付款天數選單，以及重要條款勾選框（無賠償上限、境外管轄、自動續約、智財權讓與、新設立廠商）。」
- **`B2 預測回饋與條款警示`**：
  > 「當使用者按下『立即評估合約風險』按鈕時，將輸入條件轉換為 DataFrame 餵入模型進行預測。若為高風險顯示紅色粗體並列出關鍵警告（例如：無賠償上限可能引發連帶巨額賠償），中風險顯示橘色，低風險顯示綠色並提示『可走快速免審通道』。」

---

## 📂 專案檔案清單

```text
13_legal_contract_risk/
├── AGENT.md               # 專案 Prompt：規範 AI 助教角色、字型與工程標準
├── README.md              # 本手冊：情境說明與逐步操作指示
├── lesson.ipynb           # 教學筆記本（A1～A5 實驗）
├── data/
│   ├── DATA_CARD.md       # 資料卡：欄位定義與實務意義
│   └── contract_risk.csv  # 1,000 筆合約審查歷史擬真資料
├── artifacts/
│   ├── model.joblib       # 訓練完成之 Pipeline 模型
│   └── metadata.json      # 特徵清單、標籤與模型評估分數
├── app.py                 # Tkinter 桌面預測應用程式
└── RUN.md                 # 成果驗收與操作指引
```
