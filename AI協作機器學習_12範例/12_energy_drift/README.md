# 12｜用電預測：完整專案與資料漂移

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：數值回歸與時間概念漂移（Concept Drift）監測。
- **最小應用**：Streamlit 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、時序切分與資料漂移監測規則。
- **學生專案位置**：`12_energy_drift/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可預估建物耗電並即時監測模型漂移，說出一個模型限制。

---

## 資料與問題設定

本教材原創偽資料，`data/energy_daily.csv`，600 天 × 3 棟假想建物 = 1,800 筆；固定 seed=7012（最後 120 天刻意改變耗電規則加乘 1.25）。

### 範例檔下載
- [下載 energy_daily.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/12_energy_drift/data/energy_daily.csv)：儲存至 `12_energy_drift/data/energy_daily.csv`。
- [線上資料卡](./data/DATA_CARD.md)：欄位定義與生成規則。

- **X（特徵）**：`building_type`、`floor_area_m2`、`occupancy_rate`、`temperature_forecast_c`、`is_weekend`；排除 `building_id`、`date`、`scenario_regime`、`is_synthetic`。
- **y（標籤）**：`energy_kwh`（當日用電量 kWh）。
- **切分規則**：按日期嚴格時間切分，前 360 天 train、接續 120 天 val、最後 120 天 test（漂移期封存至 A4 步驟）。絕對不可 shuffle！

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的建物耗電回歸任務，以及最後 120 天概念漂移（Concept Drift）的意義，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 12_energy_drift/ 建立 lesson.ipynb，讀取 data/energy_daily.csv。
依 AGENT.md 確認 X 與 y，檢視 3 棟建物的用電量走勢。
套用 AGENT.md 繁體中文字型設定，繪製 600 天三棟建物的總耗電時間序列圖（全繁體中文標籤）。
說明環境背景條件變化對耗電規律的潛在影響。
完成後請向我提出一個觀念思考題。
```

### A2｜按日序時間切分與特徵 Pipeline
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞工程/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
按日期順序切分：前 360 天 train、接續 120 天 val、最後 120 天 test（封存至 A4 步驟）。
建立 ColumnTransformer：
- 類別欄位（building_type）：OneHotEncoder
- 數值欄位：StandardScaler
確認同一日期的 3 棟建築嚴格位於相同分割中，不可被拆散。
完成後請向我提出一個觀念思考題。
```

### A3｜模型訓練與驗證期評估
> 觀念連結：[集成學習](../../名詞解釋/08_集成學習/README.md)、[模型參數](../../名詞解釋/05_模型參數/README.md)

```text
建立包含 ColumnTransformer 與 HistGradientBoostingRegressor / RandomForest 的 Pipeline。
在訓練集（前 360 天）上訓練模型，並在驗證集（第 361～480 天）評估 MAE 與 RMSE。
繪製驗證期「真實用電 vs 模型預測」的時間序列對照圖（全繁體中文）。
確認在規則尚未改變前，模型能良好捕捉溫度與假日的耗電規律。
完成後請向我提出一個觀念思考題。
```

### A4｜最後 120 天漂移期評估與誤差監測
> 觀念連結：[模型性能問題](../../名詞解釋/10_模型性能問題/README.md)、[機器學習訓練過程](../../名詞解釋/12_機器學習訓練過程/README.md)

```text
以在歷史正常期訓練好的模型，對最後 120 天（第 481～600 天，發生概念漂移期）test 集進行推論：
1. 計算漂移期的 MAE 與 RMSE，比較其與驗證期誤差的倍數增長（例如誤差增加 30% 以上）。
2. 計算每日預測殘差（真實值 - 預測值），並繪製 14 天滾動平均殘差趨勢圖（全繁體中文）。
3. 說明如何設定維運監測閾值（Alarm Threshold），及時發現模型失效並觸發重新訓練。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵清單、正常期 MAE、漂移期 MAE、殘差漂移警報門檻與版本。
以新 Python 程序載入模型驗證一筆建物預測。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Streamlit 介面
```text
讀取本例 Notebook 與 artifacts，使用 Streamlit 建立 app.py。
提供兩大功能分頁：
1. 「單日耗電試算」：選擇建物類型、樓地板面積、預估進駐率、氣溫預報與是否週末，預估耗電度數。
2. 「模型健康監測儀表板」：載入近期監測批次數據，視覺化呈現滾動殘差折線圖，並以紅綠燈警示目前模型是否發生概念漂移需要重新訓練。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令（streamlit run app.py）。
```

### B2｜驗收應用
```text
在介面中輸入正常日數值與極端高溫日數值，確認預測合乎邏輯。
切換至監測分頁，確認警報燈號與殘差數值與 Notebook 分析一致。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如模型只根據歷史用電規律訓練，當遭遇能源政策改變或極端氣候突變時，必須依靠監測機制觸發再訓練）。
```
