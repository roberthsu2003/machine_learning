![AI Python 機器學習實作入門](./course-cover.png)

# AI 協作機器學習｜十二個範例

這條課程路線專為學習以 AI 協作解決真實問題的學生設計。學生以自然語言提出需求，AI 建立並執行程式；學生負責看懂資料、選擇評估方式、解讀結果與驗收。每例先用 `lesson.ipynb` 理解核心概念，再把同一模型做成本機應用（Gradio 或 Streamlit）。

本課程採用現代 **專案 Prompt 驅動（Project Agent Context）** 模式：**每一個範例都是一個獨立的小專案資料夾，裡面都配置了一份專屬的 `AGENT.md`**，作為該專案的系統規範與提示詞基底。

---

## 上課方式：以 `AGENT.md` 驅動小專案實作

在過去的傳統教學中，學生需要不斷複製貼上又長又雜的提示詞（例如重複聲明中文環境、不要一次做完、固定隨機種子等）。現在，我們採用 **專案級 Prompt 規範 (`AGENT.md`)**：

```text
學生工作區/
└── 01_iris/                    # 學生為當次範例建立的小專案資料夾
    ├── AGENT.md               # 專案 Prompt：規範 AI 助教角色、繁體中文字型、防洩漏規則與任務邊界
    ├── README.md              # 學生操作手冊與逐步提示詞（A1～A5, B1～B2）
    ├── lesson.ipynb           # 由 AI 協助逐步生成的教學與實驗筆記本
    ├── data/                  # 該例所需的資料檔案
    ├── artifacts/             # 訓練完成的模型（model.joblib）與元資料（metadata.json）
    ├── app.py                 # 最終完成的本機互動 Web 應用
    └── RUN.md                 # 驗收與操作說明文件
```

### 課堂實作四步驟

1. **建立小專案資料夾**：
   學生在自己的專案目錄中，為當次範例建立資料夾（例如 `01_iris/`），並放入該範例的 `AGENT.md` 與 `README.md`。
2. **AI 自動讀取專案規範**：
   在現代 AI 開發環境（如 Antigravity、Cursor、Windsurf 或支援專案指令的工具）中開啟該資料夾，AI 會自動讀取 `AGENT.md` 作為 System Prompt；若使用網頁版 AI（ChatGPT / Claude），只需在對話一開始先附上 `AGENT.md` 內容。
3. **分段下達需求與觀念互動**：
   依序下達 `A1`～`A5`（建模實驗）與 `B1`～`B2`（Web 介面）。每完成一步，AI 會自動停下並提出一個核心觀念問題，學生確認理解後才進入下一步。
4. **驗收與產出交付**：
   完成應用驗收，重啟測試一致性，並產出 `RUN.md` 說明模型限制與啟動方式。

---

## 專案 Prompt（`AGENT.md`）應包含的核心內容

為什麼每一個範例小專案都需要 `AGENT.md`？因為它能替學生把關以下 **五大關鍵規範**：

### 1. 統一繁體中文字型與圖表繪圖規範（重點）
AI 繪圖時最常遇到中文變方塊亂碼（Tofu）、負號顯示成亂碼等問題。在 `AGENT.md` 中強制統一設定：
- **免手動下載字型**：統一使用 `mplfonts` 套件（環境預先安裝，`pip install mplfonts`），跨 Mac、Windows、Linux 與 Colab 皆通用。
- **標準字型載入程式碼**：
  ```python
  import glob, os, mplfonts
  from matplotlib import font_manager
  from mplfonts import use_font

  # 自動註冊 mplfonts 內建思源字型，並套用支援繁體中文的字型
  for f in glob.glob(os.path.join(os.path.dirname(mplfonts.__file__), "fonts", "*.[ot]tf")):
      font_manager.fontManager.addfont(f)
  use_font("Noto Sans CJK SC")  # 思源黑體，負號 unicode_minus 也會自動正確處理
  ```
- **圖表視覺呈現**：圖表標題、X/Y 座標軸、圖例說明、類別名稱一律強制使用繁體中文；資料集內部欄位保留英文字符以利程式運算，但顯示於圖表時映射為中文。

### 2. AI 助教角色與教學互動協議
- **繁體中文應答**：全程使用台灣繁體中文專業術語（例如：「特徵」、「過度擬合」、「交叉驗證」、「超參數」、「管線 Pipeline」）。
- **步驟化引導（嚴禁一口氣做完）**：嚴格禁止 AI 一次產生整份專案或整個 Notebook。每次只執行學生目前指定的單一步驟。
- **每步觀念提問**：每步完成後，AI 必須提出 1 個核心觀念思考題，引導學生思考「為什麼這樣做」，而非無腦執行。
- **誠實客觀**：不能執行時誠實告知，嚴禁捏造數據或測試分數，嚴禁為了追求高分而擅改資料。

### 3. 嚴格防資料洩漏（Anti-Data Leakage）與 ML 工程標準
- **測試集（Test Set）神聖不可侵犯**：測試集在初期切分後立即封存，僅在最後評估步驟（A4）做一次性檢驗，絕對禁止參與 EDA、Fit 或超參數挑選。
- **Pipeline 封裝**：缺失補值、特徵縮放（StandardScaler）、類別編碼（OneHotEncoder）必須嚴格包裝在 `sklearn.pipeline.Pipeline` 內，確保只在訓練折 fit。
- **可復現性**：一律固定隨機種子（如 `random_state=42`）。

### 4. 專案目錄與產出邊界限制
- **不得越界**：所有建立與修改的檔案，必須限制在當前範例資料夾內，不得污染父層或其他目錄。
- **標準產出清單**：
  - `lesson.ipynb`（步驟化的實驗 Notebook）
  - `data/`（資料集存放目錄）
  - `artifacts/model.joblib`（訓練完成的完整 Pipeline 物件）
  - `artifacts/metadata.json`（特徵、標籤、參數、指標與版本資訊）
  - `app.py`（Gradio 或 Streamlit 輕量介面）
  - `RUN.md`（啟動指引與模型限制說明）

### 5. Web 應用與模型解耦
- `app.py` 啟動時**僅允許載入已訓練完成的 `artifacts/model.joblib`**，絕對禁止在介面啟動時重新執行訓練，避免 UI 預測與 Notebook 成果不一致。

---

## 十二個範例總覽

| 編號 | 範例／資料夾 | 專案 Prompt | 資料來源 | 核心觀念 | 完成的應用 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | [鳶尾花分類](./01_iris/README.md) | [AGENT.md](./01_iris/AGENT.md) | 開放：Iris | X／y、分類、分層切分、KNN、基準 | Gradio 花種預測 |
| 02 | [企鵝種類](./02_penguins/README.md) | [AGENT.md](./02_penguins/AGENT.md) | 開放：Palmer Penguins | 缺失值、One-Hot、Pipeline、縮放 | Gradio 企鵝分類 |
| 03 | [汽車油耗](./03_auto_mpg/README.md) | [AGENT.md](./03_auto_mpg/AGENT.md) | 開放：Auto MPG | 回歸、MAE／RMSE／R²、類別特徵 | Streamlit 油耗預測 |
| 04 | [手寫數字](./04_digits/README.md) | [AGENT.md](./04_digits/AGENT.md) | 開放：Digits 子集 | 圖片特徵、分類標籤、混淆矩陣 | Gradio 8×8 測試圖辨識 |
| 05 | [葡萄酒分類](./05_wine/README.md) | [AGENT.md](./05_wine/AGENT.md) | 開放：Wine | 超參數、樹深、森林、CV、過度擬合 | Streamlit 品種分類 |
| 06 | [批發客戶分群](./06_wholesale/README.md) | [AGENT.md](./06_wholesale/AGENT.md) | 開放：Wholesale Customers | 非監督、尺度、群輪廓、投影限制 | Streamlit 客戶群輪廓 |
| 07 | [福利社營業額](./07_shop_sales/README.md) | [AGENT.md](./07_shop_sales/AGENT.md) | 偽資料：600 筆 | 權重、學習率、epoch、雜訊 | Gradio 營業額與訓練曲線 |
| 08 | [訂閱服務流失](./08_customer_churn/README.md) | [AGENT.md](./08_customer_churn/AGENT.md) | 偽資料：1,000 筆 | 不平衡、precision／recall、門檻 | Streamlit 關懷候選清單 |
| 09 | [設備維護](./09_machine_groups/README.md) | [AGENT.md](./09_machine_groups/AGENT.md) | 偽資料：50 台×20 筆 | 群組分割、洩漏、新設備泛化 | Gradio 維護關懷示範 |
| 10 | [咖啡店每日銷量](./10_cafe_time/README.md) | [AGENT.md](./10_cafe_time/AGENT.md) | 偽資料：730 天 | 時間切分、lag、只能用過去 | Streamlit 次日銷量預估 |
| 11 | [客服工單](./11_support_text/README.md) | [AGENT.md](./11_support_text/AGENT.md) | 偽資料：900 則、30 模板群 | TF-IDF、NB、模板洩漏 | Gradio 工單分流 |
| 12 | [用電與漂移](./12_energy_drift/README.md) | [AGENT.md](./12_energy_drift/AGENT.md) | 偽資料：600 天×3 棟 | 完整流程、未來泛化、誤差監測 | Streamlit 預測與批次監測 |

---

## 範例資料下載指南

環境建立由老師課堂示範，本教材從已準備好的學生專案開始。

1. **建立專案環境**：建立該範例資料夾（例如 `01_iris/`），確保資料夾內有專屬的 `AGENT.md` 與 `README.md`。
2. **下載資料（後六例及企鵝例）**：若本例有提供專用資料檔，依下方連結僅下載該例 CSV 至 `data/` 目錄，**不要下載整個 GitHub repo**。
3. **前六例開放資料**：01、04、05 由 AI 透過 scikit-learn 直接載入；02 提供下載連結；03、06 由 AI 協助載入。

### 需下載 CSV 之範例清單

| 範例 | 資料檔下載 | 存放於學生專案內位置 | 線上欄位說明 |
| :--- | :--- | :--- | :--- |
| 02 | [下載 penguins.csv](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv) | `02_penguins/data/penguins.csv` | [線上說明](https://allisonhorst.github.io/palmerpenguins/) |
| 07 | [下載 shop_sales.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/07_shop_sales/data/shop_sales.csv) | `07_shop_sales/data/shop_sales.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/07_shop_sales/data/DATA_CARD.md) |
| 08 | [下載 customer_churn.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/08_customer_churn/data/customer_churn.csv) | `08_customer_churn/data/customer_churn.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/08_customer_churn/data/DATA_CARD.md) |
| 09 | [下載 machine_readings.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/09_machine_groups/data/machine_readings.csv) | `09_machine_groups/data/machine_readings.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/09_machine_groups/data/DATA_CARD.md) |
| 10 | [下載 cafe_daily.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/10_cafe_time/data/cafe_daily.csv) | `10_cafe_time/data/cafe_daily.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/10_cafe_time/data/DATA_CARD.md) |
| 11 | [下載 support_tickets.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/11_support_text/data/support_tickets.csv) | `11_support_text/data/support_tickets.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/11_support_text/data/DATA_CARD.md) |
| 12 | [下載 energy_daily.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/12_energy_drift/data/energy_daily.csv) | `12_energy_drift/data/energy_daily.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/12_energy_drift/data/DATA_CARD.md) |

---

## 兩階段提示詞流程（A 與 B）

在有了 `AGENT.md` 作為後盾之後，學生對 AI 的操作變得非常簡潔俐落：

- **階段 A（A1～A5）：Notebook 探索與建模**
  - **A1 讀取與觀察資料**：確認特徵與標籤、檢查前五筆與缺值、繪製繁體中文探索圖表。
  - **A2 切分與前處理**：嚴格留出測試集、前處理只 fit 訓練集、防洩漏檢查。
  - **A3 訓練與比較模型**：建立基準線（Baseline）、交叉驗證（CV）、超參數調校與 Pipeline 整合。
  - **A4 評估與解讀**：以測試集進行最終評估、混淆矩陣/指標解讀、錯誤案例分析。
  - **A5 保存與確認結果**：重新執行驗證、保存完整模型（`model.joblib`）與元資料（`metadata.json`）。
- **階段 B（B1～B2）：專案應用與驗收**
  - **B1 建立介面**：讀取已保存的模型，使用 Gradio 或 Streamlit 建立 Web 介面（不重訓）。
  - **B2 驗收應用**：多組輸入測試、邊界異常檢驗、重啟測試一致性，並產出 `RUN.md`。

---

## 穿插教學：名詞解釋對照表

當學生在實作中遇到疑惑時，直接點擊對應名詞解釋連結閱讀，先開動畫與圖解，再讓 AI 結合該範例實際數字進行印證。

| 原教材章節 | 最適合回看的範例 |
| :--- | :--- |
| [01 機器學習的類型](../名詞解釋/01_機器學習的類型/README.md) | 01、03、04、06：分類、回歸、無監督 |
| [02 數據結構](../名詞解釋/02_數據結構/README.md) | 01、02、04、11：X／y、圖片、文字 |
| [03 數據分割](../名詞解釋/03_數據分割/README.md) | 01、08、09、10：分層、群組、時間序列 |
| [04 特徵工程](../名詞解釋/04_特徵工程/README.md) | 02、03、06、11：缺值、編碼、縮放、文字 |
| [05 模型參數](../名詞解釋/05_模型參數/README.md) | 05、07：樹深與權重 |
| [06 優化算法](../名詞解釋/06_優化算法/README.md) | 07：真實 epoch 與學習率 |
| [07 常見算法](../名詞解釋/07_常見算法/README.md) | 01、05、11：KNN、決策樹、Naive Bayes |
| [08 集成學習](../名詞解釋/08_集成學習/README.md) | 05、12：隨機森林、Bagging |
| [09 評估指標](../名詞解釋/09_評估指標/README.md) | 01、03、08、12：分類與回歸評估指標 |
| [10 模型性能問題](../名詞解釋/10_模型性能問題/README.md) | 05、07、12：泛化差距、過度擬合與欠擬合 |
| [11 模型泛化](../名詞解釋/11_模型泛化/README.md) | 05、09、10、12：新樣本、新設備、未來時序 |
| [12 機器學習訓練過程](../名詞解釋/12_機器學習訓練過程/README.md) | 07、10、12：訓練、保存、推論與監測 |

> [!TIP]
> 也可以直接瀏覽 [名詞解釋互動網頁](https://roberthsu2003.github.io/machine_learning/glossary/)，在課堂上直接播放動態圖解。

---

## 教師課堂引導與最終驗收標準

每完成一個範例，學生應能清楚回答：
1. **問題定義**：「我的 X 是什麼？y 是什麼？這屬於哪種類型的機器學習問題？」
2. **切分守則**：「我是怎麼分資料的？為什麼這樣分能防止偷看答案（資料洩漏）？」
3. **驗收證據**：「我用了什麼指標？基準線表現如何？我憑什麼相信這個模型有學到東西？」
4. **邊界限制**：「這個模型在什麼情況下會失效？它的限制是什麼？」

最後每位學生挑選一個完成的範例，進行 3 分鐘實機展示：展示 Web 介面輸入預測、呈現 Notebook 評估指標，並說明一個模型限制。**評量重點在於「觀念理解與成果驗收」，不考死背 Python 語法**。

[返回主教材](../README.md)
