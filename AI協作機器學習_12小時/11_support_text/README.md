# 11｜客服工單：文字也能建立模型

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：繁體中文文字分類（客服工單意圖分流）。
- **最小應用**：Gradio 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、模板群切分（GroupShuffleSplit）與防文字洩漏規則。
- **學生專案位置**：`11_support_text/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可輸入任意顧客諮詢文字自動分流，並說出一個模型限制。

---

## 資料與問題設定

本教材原創偽資料，`data/support_tickets.csv`，900 則繁體中文客服工單、30 個模板群；固定 seed=7011。

### 範例檔下載
- [下載 support_tickets.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/11_support_text/data/support_tickets.csv)：儲存至 `11_support_text/data/support_tickets.csv`。
- [線上資料卡](./data/DATA_CARD.md)：欄位定義與生成規則。

- **X（特徵）**：只用 `text` 欄位（繁體中文文字）；排除 `ticket_id`、`template_group`、`is_synthetic`。
- **y（標籤）**：`intent`（三類：配送查詢／退換貨／付款問題）。
- **切分規則**：使用 `GroupShuffleSplit` 依據 `template_group` 切分 60/20/20 train/val/test；相同模板文字絕對不跨組！

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的文字特徵（TF-IDF）提取方式，以及為什麼客服模板若跨組會造成虛假高分，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 11_support_text/ 建立 lesson.ipynb，讀取 data/support_tickets.csv。
依 AGENT.md 確認 X（text）與 y（intent），顯示前 5 筆訊息與三個意圖類別的分佈比例。
套用 AGENT.md 繁體中文字型設定，繪製各意圖的字數長度分佈直方圖（全繁體中文標籤）。
說明真實客服訊息中常見的贅字、同義詞與語氣變化。
完成後請向我提出一個觀念思考題。
```

### A2｜模板群切分（GroupShuffleSplit）與 TF-IDF Pipeline
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞工程/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
1. 使用 GroupShuffleSplit 依 template_group 切分出 60% train、20% val、20% test（封存至 A4 步驟）。
2. 驗證 train/val/test 的模板群集合完全互斥、交集為空。
3. 建立包含 TfidfVectorizer(max_features=1000) 的 Pipeline，說明為什麼 fit 只能在 train 上進行。
完成後請向我提出一個觀念思考題。
```

### A3｜模型訓練與比較
> 觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)

```text
以 DummyClassifier（多數決）為基準線。
在開發集上比較 MultinomialNB（單純貝氏）與 LinearSVC（線性支援向量機）的 macro-F1。
在驗證集上評估，選出表現最佳的模型 Pipeline。
列出各意圖類別中 TF-IDF 權重最高的前 5 個代表性關鍵詞（全繁體中文）。
完成後請向我提出一個觀念思考題。
```

### A4｜測試集最終評估
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以最佳 Pipeline 對封存的全新模板 test 集進行一次性最終評估。
顯示 test accuracy、macro-F1 與繁體中文混淆矩陣。
挑出一則被模型分類錯誤的工單文本，分析其是否包含多種意圖的重疊語句。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整包含 TfidfVectorizer 與分類器的 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄字彙量上限、意圖標籤對應與測試集評估指標。
以新 Python 程序載入模型驗證一句全新手打工單句子（例如：「我上週刷卡付費成功，但訂單狀態怎麼還是顯示待付款？」）。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Gradio 介面
```text
讀取本例 Notebook 與 artifacts，使用 Gradio 建立 app.py。
提供文字輸入大文字框（預先提供「包裹卡在海關」、「想退換貨尺寸不合」、「重複扣款問題」等範例按鈕）。
點擊「工單意圖分流」後輸出預測的類別與三大意圖的信心度長條圖。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令。
```

### B2｜驗收應用
```text
輸入 3 句不同長度的自然語言測試句，確認分流預測與 Notebook 一致。
輸入一句完全無關的問候語（例如：「今天天氣真好」），觀察模型反應並記錄其限制。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如基於詞袋與統計模型，對語意反諷或未見詞彙較無深層理解能力）。
```
