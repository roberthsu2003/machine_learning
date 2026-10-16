# 08｜訂閱服務流失：不平衡與預警門檻

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：不平衡二元分類（預測訂閱用戶流失）。
- **最小應用**：Streamlit 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、不平衡評估指標與防洩漏規則。
- **學生專案位置**：`08_customer_churn/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可動態調節預警門檻篩選關懷名單，並說出一個模型限制。

---

## 資料與問題設定

本教材原創偽資料，`data/customer_churn.csv`，1,000 筆；固定 seed=7008。

### 範例檔下載
- [下載 customer_churn.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E7%AF%84%E4%BE%8B/08_customer_churn/data/customer_churn.csv)：儲存至 `08_customer_churn/data/customer_churn.csv`。
- [線上資料卡](./data/DATA_CARD.md)：欄位定義與生成規則。

- **X（特徵）**：用戶使用天數、月費、客服進線次數等；排除 `customer_id`、`is_synthetic`。
- **y（標籤）**：`churn`（0: 續約, 1: 流失；正例約 15%～20% 不平衡）。
- **切分規則**：使用 `stratify=y`、`random_state=42` 切分 60/20/20 train/val/test；測試集封存至 A4 步驟。

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的不平衡二元分類特質、為什麼不能只看準確率（Accuracy），等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 08_customer_churn/ 建立 lesson.ipynb，讀取 data/customer_churn.csv。
依 AGENT.md 確認 X／y，計算正負樣本比例（流失 vs 續約）。
套用 AGENT.md 繁體中文字型設定，繪製流失比例長條圖與關鍵特徵盒鬚圖（全繁體中文標籤）。
說明如果模型全部猜「不流失」，準確率看似很高但有什麼致命問題（準確率陷阱）。
完成後請向我提出一個觀念思考題。
```

### A2｜切分與前處理 Pipeline
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞工程/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
使用分層切分（stratify=y, random_state=42）將資料切分為 60% train、20% val 與 20% test。
建立包含補值與特徵縮放的 ColumnTransformer。
說明在不平衡資料下，為什麼切分時「stratify」格外不可或缺。
完成後請向我提出一個觀念思考題。
```

### A3｜模型訓練與分類門檻（Threshold）調校
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以 LogisticRegression(class_weight='balanced') 與 RandomForestClassifier 訓練模型。
在驗證集（val）上計算各門檻（0.1 到 0.9）下的 Precision、Recall 與 F1-score。
繪製 Precision-Recall 曲線圖與不同門檻下的混淆矩陣變化（全繁體中文）。
根據商業情境（挽留每位客戶的成本 vs 漏報流失客戶的損失）選擇最佳門檻（如 0.35）。
完成後請向我提出一個觀念思考題。
```

### A4｜測試集最終評估
> 觀念連結：[模型性能問題](../../名詞解釋/10_模型性能問題/README.md)

```text
以選好的最佳模型與門檻，對封存的 test 集進行一次性最終評估。
顯示 test Precision、Recall、F1-score 與繁體中文混淆矩陣。
與全部猜 0 的基準線比較，計算模型成功多抓出多少潛在流失客戶。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵清單、正例比例、推薦預警門檻（threshold）與測試集各指標。
以新 Python 程序載入模型驗證一名高風險客戶的機率預測。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Streamlit 介面
```text
讀取本例 Notebook 與 artifacts，使用 Streamlit 建立 app.py。
提供用戶基本屬性與行為數據的輸入面板，並在側邊欄提供「流失預警敏感度門檻（0.1～0.9）」的互動滑桿。
動態輸出流失機率、風險等級（低/中/高）與是否需要立即發送關懷優惠券的建議。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令（streamlit run app.py）。
```

### B2｜驗收應用
```text
調整滑桿門檻，確認風險標籤與關懷建議隨門檻動態變化，並與 Notebook 一致。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如該模型基於假想固定資費用戶行為，若公司推出全新產品方案可能發生行為改變）。
```
