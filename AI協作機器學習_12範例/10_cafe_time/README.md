# 10｜咖啡店：只能用過去預測未來

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：時間序列回歸（次日咖啡銷量預估）。
- **最小應用**：Streamlit 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、時序切分（不 shuffle）與防前瞻洩漏規則。
- **學生專案位置**：`10_cafe_time/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可以昨日銷量與今日預報推估當日杯數，並說出一個模型限制。

---

## 資料與問題設定

本教材原創偽資料，`data/cafe_daily.csv`，2024-01-01 起連續 730 天；固定 seed=7010。

### 範例檔下載
- [下載 cafe_daily.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/10_cafe_time/data/cafe_daily.csv)：儲存至 `10_cafe_time/data/cafe_daily.csv`。
- [線上資料卡](./data/DATA_CARD.md)：欄位定義與生成規則。

- **X（特徵）**：`day_of_week`、`promo_planned`、`temperature_forecast_c`、`rain_forecast`，以及由歷史銷量衍生出的 `sales_lag_1` 與 `sales_rolling_7`（rolling 必須先 shift(1)）。
- **y（標籤）**：`sales_cups`（當日銷量杯數）；嚴禁使用當日實際銷量或未來天氣預測。
- **切分規則**：按時間嚴格順序，前 510 天 train、接續 110 天 val、最後 110 天 test；絕對不可 shuffle！

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的時間序列特性，以及為什麼時序資料絕對不能打亂（shuffle）切分，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 10_cafe_time/ 建立 lesson.ipynb，讀取 data/cafe_daily.csv。
確認 date 為時間序列索引，檢視前 10 天與後 10 天的資料。
套用 AGENT.md 繁體中文字型設定，繪製 730 天每日咖啡銷量走勢折線圖（全繁體中文標籤）。
說明時序資料中「時間箭頭不可逆」的根本特性。
完成後請向我提出一個觀念思考題。
```

### A2｜特徵工程（Lag 與 Rolling）與時序順序切分
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞工程/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則建構特徵與切分：
1. 建立昨日銷量 sales_lag_1 = df['sales_cups'].shift(1)。
2. 建立過去 7 天平均 sales_rolling_7 = df['sales_cups'].shift(1).rolling(7).mean()。
3. 移除前 7 天產生的 NaN，依日期順序切分：
   - Train：日序 0～509 天
   - Val：日序 510～619 天
   - Test：日序 620～729 天（封存至 A4 步驟）
4. 說明為什麼 rolling 前一定要先 shift(1)（防前瞻偏差 Lookahead Bias）。
完成後請向我提出一個觀念思考題。
```

### A3｜模型訓練與基準線比較
> 觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)、[模型參數](../../名詞解釋/05_模型參數/README.md)

```text
以「天真預測基準線（Naive Baseline：猜明天銷量等於昨天銷量）」作為基準。
在訓練集訓練 Ridge 回歸與 GradientBoostingRegressor（梯度提升樹）。
在驗證集（val）評估預測表現，比較 MAE、RMSE 與 WAPE（加權絕對百分比誤差）。
解釋比較結果並選出最佳模型，完成後請向我提出一個觀念思考題。
```

### A4｜未來 110 天測試集最終評估
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)、[模型泛化](../../名詞解釋/11_模型泛化/README.md)

```text
以最佳模型對封存的未來 110 天 test 集進行一步預測（One-step ahead prediction）評估。
計算 test MAE、RMSE。
繪製最後 110 天「真實銷量 vs 模型次日預估」的疊加折線圖（全繁體中文）。
挑出一個預測大幅偏離的日子（例如暴雨或意外促銷），分析其成因。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與時序回歸模型保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵清單、時序切分日期、落後特徵定義與未來 110 天評估指標。
以新 Python 程序載入模型驗證次日銷量預估。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Streamlit 介面
```text
讀取本例 Notebook 與 artifacts，使用 Streamlit 建立 app.py。
提供店長每日開店前輸入介面：昨日實際銷量杯數、過去一週平均、今日星期幾、是否安排促銷、今日氣溫與降雨預報。
點擊「預估今日備料杯數」後輸出預估值與上下 80% 信心區間。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令（streamlit run app.py）。
```

### B2｜驗收應用
```text
輸入 3 種不同情境（例如晴天促銷日 vs 寒流暴雨日），確認杯數預測合乎直覺與模型特性。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如該模型是一步預測模型，無法直接一次預測未來一個月後的每日銷量）。
```
