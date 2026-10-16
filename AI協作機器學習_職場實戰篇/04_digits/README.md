# 04｜手寫數字：圖片也是特徵

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。

- **任務**：多類別影像分類（手寫數字 0～9）。
- **最小應用**：Gradio 網頁應用。
- **專案 Prompt**：本資料夾內已備妥 [AGENT.md](./AGENT.md)，規範 AI 助教角色、繁體中文字型、混淆矩陣解讀與防洩漏規則。
- **學生專案位置**：`04_digits/`（共用專案根目錄的 `.venv`）。
- **本例完成後**：可透過測試圖辨識手寫數字，並說出一個模型限制。

---

## 資料與問題設定

[UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)，採 sklearn `load_digits` 的 1,797 筆 8×8 灰階子集；CC BY 4.0。A1 由 AI 載入並儲存在本例 `data/`。

- **X（特徵）**：64 個像素強度（0～16），按照固定順序展開為一維向量。
- **y（標籤）**：阿拉伯數字 0～9 的類別（分類任務，非數值回歸）。
- **切分規則**：以 `stratify=y`、`random_state=42` 切分 80/20；測試集嚴格封存至 A4 步驟，並保留未訓練的 test 圖片供 app 展示。

---

## 上課流程：AI 專案 Prompt 協作

在開始前，請確認 AI 已讀取當前資料夾的 [AGENT.md](./AGENT.md)。

> **啟動指令（貼給 AI 的第一句話）**：
> ```text
> 請詳細閱讀本資料夾的 AGENT.md，擔任我的機器學習助教，嚴格遵守繁體中文、字型設定、單步執行與防洩漏規則。
> 現在請簡要向我說明本專案的圖片特徵（64 像素）如何展開、分類標籤 y，等待我給予下一步指令。
> ```

---

## A｜一步一步建立 Notebook（lesson.ipynb）

每次只貼一段 prompt，AI 執行後會提出一個觀念問題，確認理解後再貼下一段。

### A1｜讀取與觀察資料
> 觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
請在 04_digits/ 建立 lesson.ipynb，載入 sklearn 的 load_digits 資料集。
依 AGENT.md 確認 X／y，檢視資料維度與標籤分佈。
套用 AGENT.md 繁體中文字型設定，以 8×8 網格繪製前 10 張數字影像與對應繁體中文標籤。
說明電腦如何將一張圖片看作 64 個數值特徵，完成後請向我提出一個觀念思考題。
```

### A2｜切分與前處理 Pipeline
> 觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[特徵工程](../../名詞工程/04_特徵工程/README.md)

```text
依 AGENT.md 切分規則切分資料：
80/20 分層切分（stratify=y, random_state=42），封存 test 集。
將像素強度正規化（縮放至 0～1）放入 Pipeline。
列出 train 與 test 的筆數，確認無資料洩漏。
完成後請向我提出一個觀念思考題。
```

### A3｜訓練與比較模型
> 觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)、[模型參數](../../名詞解釋/05_模型參數/README.md)

```text
以 DummyClassifier 為基準線。
比較 LogisticRegression 與 SVC (Support Vector Machine) 在開發集 3 折 CV 下的 macro-F1。
嚴格遵守：所有特徵縮放均在 CV 折內完成，不碰測試集。
解釋比較結果並選出最佳模型，完成後請向我提出一個觀念思考題。
```

### A4｜評估與解讀
> 觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)

```text
以最佳模型對封存的 test 集進行一次性最終評估。
顯示 test accuracy、macro-F1 與 10×10 繁體中文混淆矩陣熱力圖。
找出最容易被模型看錯的兩個數字組合（例如 8 和 1，或 3 和 5），並視覺化繪出被誤判的實際影像。
完成後請向我提出一個觀念思考題。
```

### A5｜保存與確認結果
```text
重啟 kernel 從頭重新執行 lesson.ipynb，確保圖表繁中與結果可復現。
將完整前處理與模型 Pipeline 保存至 artifacts/model.joblib。
另存 artifacts/metadata.json：記錄特徵長度、最佳模型與測試集評估指標。
同時在 artifacts/ 另外保存 5 張來自測試集的範例圖片陣列（供 Web 介面測試使用）。
完成後回報成果。
```

---

## B｜把模型做成應用（app.py）

完成 A5 後再進行 B1、B2。

### B1｜建立 Gradio 介面
```text
讀取本例 Notebook 與 artifacts，使用 Gradio 建立 app.py。
提供手繪畫布（Sketchpad）或從下拉選單選擇測試集樣本圖片。
點擊「預測」按鈕後，顯示模型辨識出的數字以及各數字（0～9）的預測機率長條圖。
嚴格守則：直接 joblib.load 載入 artifacts/model.joblib，絕對不在 app.py 內重新訓練！
介面文字全繁體中文，標示輸入限制。提供本機啟動命令。
```

### B2｜驗收應用
```text
測試 3 張清晰的數字圖片與 1 張筆劃模糊的圖片，確認辨識結果與機率分佈合理。
產出 RUN.md，說明：如何啟動應用、如何操作、以及本模型的一項已知局限性（例如該模型專門訓練於 8×8 低解析度置中黑白數字，若輸入高解析度或傾斜數字將會失效）。
```
