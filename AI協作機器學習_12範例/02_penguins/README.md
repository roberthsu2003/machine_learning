# 02｜企鵝：缺失值與類別欄位

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：多類別分類（三種企鵝）。
- **最小應用**：Gradio 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、缺失值處理與 Pipeline 防洩漏規則。
- **學生專案位置**：`02_penguins/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可用原始輸入（含島嶼與性別）執行預測，並說出一個模型限制。

---

## 資料與問題設定

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)，344 筆、3 種企鵝；CC0。

### 範例檔下載
[下載 penguins.csv](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv)，儲存到學生專案的 `02_penguins/data/penguins.csv`。

- **X（特徵）**：
  - 數值：`bill_length_mm`、`bill_depth_mm`、`flipper_length_mm`、`body_mass_g`。
  - 類別：`island`、`sex`。
  - 排除：`year` 不納入本例特徵；禁止將 `species` 納入 X。
- **y（標籤）**：`species`（Adelie / Chinstrap / Gentoo）。
- **切分規則**：先分層切分 80/20（`stratify=y, random_state=42`），再對訓練集 fit 缺失補值與編碼；測試集嚴格封存至 A4 步驟。

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的數值與類別特徵、標籤 y，以及缺失值處理策略，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 02_penguins/ 建立 lesson.ipynb，優先讀取 data/penguins.csv。
依 AGENT.md 確認 X／y，檢視資料形狀、前五筆，並統計各欄位的缺失值比例。
套用 AGENT.md 繁體中文字型設定，繪製各特徵與企鵝種類的關係圖（全繁體中文標籤）。
說明只有 y 缺失時才能刪列，特徵缺失應如何透過 Pipeline 補值。
完成後請向我提出一個觀念思考題。
```

### A2｜切分與前處理 Pipeline
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞解釋/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
1. 移除 y 為空的列，以 80/20 分層切分（stratify=y, random_state=42），封存 test 集。
2. 建立 ColumnTransformer：
   - 數值欄位：SimpleImputer(strategy='median') ＋ StandardScaler
   - 類別欄位：SimpleImputer(strategy='most_frequent') ＋ OneHotEncoder(handle_unknown='ignore')
3. 說明為什麼「先切分再補值」是防止資料洩漏的鐵律。
完成後請向我提出一個觀念思考題。
```

### A3｜訓練與比較模型
> 觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)、[模型參數](../../名詞解釋/05_模型參數/README.md)

```text
以 DummyClassifier 為基準線。
將上述 ColumnTransformer 與 LogisticRegression 或 SVC 結合成完整 Pipeline。
在開發集進行 3 折分層交叉驗證，比較不同模型或超參數的 macro-F1。
嚴格遵守：所有前處理都在 CV 折內由 Pipeline 完成，不碰測試集。
解釋比較結果並選出最佳設定，完成後請向我提出一個觀念思考題。
```

### A4｜評估與解讀
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以最佳 Pipeline 對封存的 test 集進行一次性最終評估。
顯示 test accuracy、macro-F1、分類報告與繁體中文混淆矩陣。
與基準線比較，並挑出一個可能混淆的案例進行生物特徵上的合理性分析。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄欄位類型、類別對應、補值中位數/眾數、最佳超參數與測試集指標。
以新 Python 程序載入保存的模型，驗證一筆包含缺值的測試樣本，確認能正確推論。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Gradio 介面
```text
讀取本例 Notebook 與 artifacts，使用 Gradio 建立 app.py。
提供 4 個數值特徵的滑桿/輸入框，以及島嶼（island）與性別（sex）的下拉選單。
點擊後輸出預測的企鵝種類與各品種機率。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令。
```

### B2｜驗收應用
```text
測試 3 組有效輸入（包含不同島嶼與性別組合），確認 app.py 與 Notebook 預測結果一致。
測試邊界數值，確認缺少性別輸入時 Pipeline 是否能妥善容錯。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如該模型適用於帕爾默群島樣本，其他地區企鵝體型比例可能不同）。
```
