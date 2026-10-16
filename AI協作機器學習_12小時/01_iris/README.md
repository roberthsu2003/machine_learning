# 01｜鳶尾花：第一次讓 AI 完成分類

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：多類別分類（三種鳶尾花）。
- **最小應用**：Gradio 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型與防洩漏規則。
- **學生專案位置**：`01_iris/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可用原始輸入（4 個 cm 特徵）執行預測，並說出一個模型限制。

---

## 資料與問題設定

[UCI Iris](https://archive.ics.uci.edu/dataset/53/iris)，150 筆、4 特徵、3 類；CC BY 4.0。作者 R. A. Fisher，DOI：10.24432/C56C76。
本例使用 scikit-learn 內附的開放資料，A1 會由 AI 載入並儲存在本例 `data/`。

- **X（特徵）**：`sepal length`、`sepal width`、`petal length`、`petal width`（單位均為 cm）；target 不可放入 X。
- **y（標籤）**：三種鳶尾花類別（Setosa / Versicolour / Virginica）。
- **切分規則**：先以 `stratify=y`、`random_state=42` 留出 20% test 作為最終驗收；80% 開發集內部做 3 折 `StratifiedKFold`，test 只在選型完成後使用一次。

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的任務類型、特徵 X 與標籤 y，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 01_iris/ 建立 lesson.ipynb，載入 sklearn 的 load_iris 資料。
依 AGENT.md 設定確認 X／y，顯示前五筆、欄位與缺失值統計。
套用 AGENT.md 繁體中文字型設定，繪製特徵分佈或散佈圖（全繁體中文標籤），並以生活化語言說明一列資料代表什麼。
完成後請向我提出一個觀念思考題。
```

### A2｜切分與前處理
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞解釋/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
80/20 分層切分（stratify=y, random_state=42），封存 test 集。
說明為什麼此處需要分層切分（stratify）。
列出 train 與 test 的筆數與各類別分佈，檢查有無重疊與洩漏。
完成後請向我提出一個觀念思考題。
```

### A3｜訓練與比較模型
> 觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)、[模型參數](../../名詞解釋/05_模型參數/README.md)

```text
以 DummyClassifier（多數決）作為基準線（Baseline）。
建立 StandardScaler ＋ KNeighborsClassifier 的 Pipeline。
在開發集上以 3 折分層交叉驗證（StratifiedKFold），比較 k＝3、5、7 的 macro-F1，並比較有無特徵縮放（Scaler）的差異。
比較時只能用開發集 CV，絕對不能使用 test 集調參。
解釋比較結果並選出最佳設定，完成後請向我提出一個觀念思考題。
```

### A4｜評估與解讀
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以選好的最佳模型 Pipeline，對封存的 test 集進行一次性最終評估。
顯示 test accuracy、macro-F1 與繁體中文混淆矩陣。
與基準線比較，並挑出一個預測錯誤案例（若無錯誤則說明最接近邊界的樣本）進行原因分析。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵欄位、目標名稱、切分設定、最佳超參數、評估指標與版本。
以新 Python 程序載入保存的模型，驗證一組輸入預測，確認與 Notebook 一致。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Gradio 介面
```text
讀取本例 Notebook 與 artifacts，使用 Gradio 建立 app.py。
提供 4 個 cm 特徵的數值輸入元件（預設填入合理的平均值），點擊後輸出預測的花種與各類別機率。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令。
```

### B2｜驗收應用
```text
測試 3 組有效輸入，確認 app.py 與 Notebook 預測結果一致。
測試邊界與極端數值輸入，確認防呆機制。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如受限於 1936 年特定樣本，無法直接推廣至所有野生變種）。
```
