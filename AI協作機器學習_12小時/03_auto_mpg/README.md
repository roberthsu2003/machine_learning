# 03｜汽車油耗：預測數字與誤差

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：連續數值回歸（預測油耗效率 mpg）。
- **最小應用**：Streamlit 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、回歸評估指標與防洩漏規則。
- **學生專案位置**：`03_auto_mpg/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可用車輛規格預測油耗數字，並說出一個模型限制。

---

## 資料與問題設定

[UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg)，398 筆；CC BY 4.0。A1 由 AI 從 UCI 來源快取至 `03_auto_mpg/data/`。

- **X（特徵）**：
  - 數值：`cylinders`、`displacement`、`horsepower`、`weight`、`acceleration`、`model_year`。
  - 類別：`origin`（產地代碼 1: 美國, 2: 歐洲, 3: 日本）。
  - 排除：`car_name`、車輛 ID 與預測目標 `mpg`。
  - 缺失：`horsepower` 欄位存在缺失值，由 Pipeline 中位數補值處理。
- **y（標籤）**：`mpg`（每美制加侖英里數，數值越大越省油；不可稱為分類準確率）。
- **切分規則**：固定 `random_state=42` 隨機 80/20 切分；測試集嚴格封存至 A4 步驟。

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的回歸任務目標、特徵欄位與評估指標（MAE / RMSE / R²），等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 03_auto_mpg/ 建立 lesson.ipynb，載入 Auto MPG 資料集。
依 AGENT.md 確認 X／y，檢視資料形狀、前五筆，並找出缺失值（如 horsepower）。
套用 AGENT.md 繁體中文字型設定，繪製特徵（如馬力、車重）與油耗 mpg 的散佈圖（全繁體中文標籤）。
說明一列資料代表什麼，完成後請向我提出一個觀念思考題。
```

### A2｜切分與前處理 Pipeline
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞解釋/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
80/20 隨機切分（random_state=42），封存 test 集。
建立 ColumnTransformer：
- 數值欄位：SimpleImputer(strategy='median') ＋ StandardScaler
- 類別欄位（origin）：OneHotEncoder(handle_unknown='ignore')
列出 train 與 test 的筆數，確認無資料洩漏。
完成後請向我提出一個觀念思考題。
```

### A3｜訓練與比較模型
> 觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)、[模型參數](../../名詞解釋/05_模型參數/README.md)

```text
以 DummyRegressor（平均值預測）作為基準線。
建立 Ridge 回歸與 RandomForestRegressor 的 Pipeline。
在開發集進行 3 折交叉驗證，比較負均方誤差（neg_mean_squared_error）與 R²。
嚴格遵守：所有前處理都在 CV 折內由 Pipeline 完成，不碰測試集。
解釋比較結果並選出最佳設定，完成後請向我提出一個觀念思考題。
```

### A4｜評估與解讀
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以最佳 Pipeline 對封存的 test 集進行一次性最終評估。
計算並顯示 test MAE、RMSE 與 R²。
與基準線比較，並繪製「真實油耗 vs 預測油耗」的散佈圖與殘差圖（全繁體中文）。
挑出一個預測誤差較大的案例進行原因分析。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵欄位、產地對應、最佳模型參數與測試集評估指標（MAE, RMSE, R²）。
以新 Python 程序載入保存的模型，驗證一輛假想車的油耗預測，確認與 Notebook 一致。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Streamlit 介面
```text
讀取本例 Notebook 與 artifacts，使用 Streamlit 建立 app.py。
提供氣缸數、排氣量、馬力、車重、加速度、年份與產地的輸入元件。
點擊按鈕後顯示預測的油耗 MPG，並換算為生活熟悉的公制單位（公里/公升）。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令（streamlit run app.py）。
```

### B2｜驗收應用
```text
測試 3 組有效輸入（如美系大馬力車 vs 日系輕量車），確認 app.py 與 Notebook 預測結果一致。
檢查輸入邊界與異常值提示。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如該模型基於 1970～1980 年代老車資料，無法直接套用於現代油電車或電動車）。
```
