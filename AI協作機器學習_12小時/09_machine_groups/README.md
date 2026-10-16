# 09｜設備保養：同一台機器不能偷看考卷

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：群組分割與二元分類（設備故障預警）。
- **最小應用**：Gradio 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、群組切分（GroupShuffleSplit）與防洩漏規則。
- **學生專案位置**：`09_machine_groups/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可為未曾見過的新設備評估故障機率，並說出一個模型限制。

---

## 資料與問題設定

本教材原創偽資料，`data/machine_readings.csv`，50 台設備各 20 筆觀測，共 1,000 筆；固定 seed=7009。

### 範例檔下載
- [下載 machine_readings.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/09_machine_groups/data/machine_readings.csv)：儲存至 `09_machine_groups/data/machine_readings.csv`。
- [線上資料卡](./data/DATA_CARD.md)：欄位定義與生成規則。

- **X（特徵）**：`temperature_c`、`vibration_mm_s`、`operating_hours`、`machine_age_years`；排除 `machine_id`、`reading_id`、`is_synthetic`。
- **y（標籤）**：`failure_next_7d`（未來 7 天是否故障，0 或 1）。
- **切分規則**：使用 `GroupShuffleSplit` 依 `machine_id` 切分；30 台 train、10 台 val、10 台 test，三組設備 ID 絕對不可重疊！

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的群組分割概念，以及為什麼「同一台機器的讀數不能拆在 train 和 test」，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 09_machine_groups/ 建立 lesson.ipynb，讀取 data/machine_readings.csv。
依 AGENT.md 確認 X、y 與群組欄位 machine_id。
統計每台機器的讀數筆數（每台 20 筆）與整體故障率。
套用 AGENT.md 繁體中文字型設定，繪製不同設備的溫度與振動分佈盒鬚圖（全繁體中文標籤）。
說明同一台機器的不同時間讀數彼此高度相關的物理現象。
完成後請向我提出一個觀念思考題。
```

### A2｜群組切分（GroupShuffleSplit）防洩漏
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞工程/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
1. 使用 GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 依 machine_id 留出 10 台機器作為 test 集封存。
2. 對其餘 40 台再以 GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43) 留出 10 台作為 val 集。
3. 嚴格驗證：使用集合運算（Intersection）確認 train、val、test 三個集合的 machine_id 完全互斥、重疊數為 0！
完成後請向我提出一個觀念思考題。
```

### A3｜模型訓練與隨機切分洩漏對照實驗
> 觀念連結：[模型泛化](../../名詞解釋/11_模型泛化/README.md)

```text
進行一次震撼對照實驗：
1. 如果錯誤地使用「一般隨機切分（Random Split）」，讓同一台機器的資料被打散在 train 和 test，觀察模型在 test 上的成績（虛假高分）。
2. 在正確的「群組切分（GroupSplit）」下訓練 RandomForestClassifier，評估模型在全新 10 台未見設備上的真實表現。
繪製長條圖比較兩種切分方式的評估差距，並說明何謂「群組資料洩漏（Group Leakage）」。
完成後請向我提出一個觀念思考題。
```

### A4｜測試集最終評估
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以在群組切分下訓練的最佳模型，對封存的全新 10 台設備 test 集進行一次性最終評估。
顯示 test ROC-AUC、PR-AUC、Precision、Recall 與繁體中文混淆矩陣。
挑出一台剛出廠但被誤判有故障風險的設備，分析其感測讀數特徵。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵欄位、群組切分設備列表、隨機種子與對未見新設備的測試指標。
以新 Python 程序載入模型驗證一台全新假想設備的讀數推論。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Gradio 介面
```text
讀取本例 Notebook 與 artifacts，使用 Gradio 建立 app.py。
提供運轉溫度、振動幅度、累積運轉時數、設備年限 4 個數值輸入框（標註警示範圍）。
點擊「評估設備健康狀態」後，輸出該設備在未來 7 天的故障風險機率與保養建議指示燈（綠色正常/黃色注意/紅色即刻停機維護）。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令。
```

### B2｜驗收應用
```text
測試 3 組全新設備感測值（包含正常機台與即將磨損機台），確認 app.py 與 Notebook 預測結果一致。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如模型假設新設備屬於相同製造批次規格，若工廠引進全新品牌型號設備，其基線溫度與振動基準可能完全不同）。
```
