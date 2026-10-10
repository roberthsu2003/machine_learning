# 名詞解釋：12 章互動動畫教材

本教材保留「名詞解釋」既有的 01–12 章順序與四大模組。教材仍以原章節 README 為主，每章新增一個獨立網址，供教師從 GitHub 進入對應的互動動畫教學頁。

已完成 12 個章節頁、總目錄與 42 個 Canvas 2D 互動場景。所有頁面採 Light Mode，保留原教材小節順序、表格與公式，原圖可展開對照。推送到 GitHub Pages 使用的分支後即可發布。

## 子目錄

```text
docs/
├── index.html                         # 既有機器學習簡報網頁
├── course.css / course.js             # 既有教材資源
├── motion.css / motion.js             # 既有簡報 2D 動畫
└── glossary/
    ├── index.html                    # 12 章總目錄
    ├── README.md                     # 教材與維護說明
    ├── chapters.json                 # 章序、來源、網址與場景清單
    ├── shared/                       # 共用樣式、動畫播放器與幾何工具
    ├── 01-learning-paradigms/
    ├── 02-data-structures/
    ├── 03-data-splitting/
    ├── 04-feature-engineering/
    ├── 05-model-parameters/
    ├── 06-optimization/
    ├── 07-common-algorithms/
    ├── 08-ensemble-learning/
    ├── 09-evaluation-metrics/
    ├── 10-model-performance/
    ├── 11-generalization/
    └── 12-training-process/
```

每章已有 `README.md`，記錄本章的動畫設計與原圖對應。每章採用下列配置：

```text
06-optimization/
├── README.md
├── index.html
├── lesson.js
├── scenes/
│   └── index.js                      # 本章所有場景定義
└── assets/
    └── images/                       # 發布時需用到的原教材圖片
```

每章使用 `index.html`，網址即可省略檔名。英文路徑方便分享與維護，頁面標題與所有教學說明使用繁體中文。

## 章序與固定網址

網址基底：`https://roberthsu2003.github.io/machine_learning/glossary/`。

| 原章序 | 章節 | 子目錄／網址尾段 | 幾何動畫重點 | 場景說明 |
| :---: | :--- | :--- | :--- | :--- |
| 01 | 學習範式 | `01-learning-paradigms/` | 帶標籤／無標籤資料流、分類與回歸、聚類與投影 | [3 個場景](01-learning-paradigms/README.md) |
| 02 | 數據結構 | `02-data-structures/` | 欄位方塊拆解、標籤移出、X 矩陣與 y 向量重排 | [3 個場景](02-data-structures/README.md) |
| 03 | 數據分割 | `03-data-splitting/` | 資料方塊分箱、分層與時間切分、洩漏路徑 | [4 個場景](03-data-splitting/README.md) |
| 04 | 特徵工程 | `04-feature-engineering/` | 特徵淡出、縮放後散點移動、One-Hot 方塊展開 | [3 個場景](04-feature-engineering/README.md) |
| 05 | 模型參數 | `05-model-parameters/` | 權重改變線條、超參數旋鈕、搜尋點逐次亮起 | [3 個場景](05-model-parameters/README.md) |
| 06 | 優化算法 | `06-optimization/` | 小球沿損失曲線下降、步長震盪、梯度路徑 | [4 個場景](06-optimization/README.md) |
| 07 | 常見算法 | `07-common-algorithms/` | KNN 距離投票、決策樹切割、SVM 間隔、機率更新 | [4 個場景](07-common-algorithms/README.md) |
| 08 | 集成學習 | `08-ensemble-learning/` | 並行投票、逐輪補殘差、OOF 預測匯入元模型 | [3 個場景](08-ensemble-learning/README.md) |
| 09 | 評估指標 | `09-evaluation-metrics/` | 四象限落格、門檻與 ROC、殘差平方塊、R² 對照 | [4 個場景](09-evaluation-metrics/README.md) |
| 10 | 模型性能問題 | `10-model-performance/` | 模型曲線變形、重抽樣波動、學習曲線與早停 | [4 個場景](10-model-performance/README.md) |
| 11 | 模型泛化 | `11-generalization/` | 新樣本投入、誤差鴻溝、K 折輪替與改善策略 | [4 個場景](11-generalization/README.md) |
| 12 | 機器學習訓練過程 | `12-training-process/` | 六階段閉環、訓練迴圈、Batch／Iteration／Epoch | [3 個場景](12-training-process/README.md) |

例如第 06 章的完整網址為：

```text
https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/
```

## 從 GitHub 進入教學頁

每章原 README 的標題下方已有「開啟本章互動動畫教學」與場景錨點連結；總 README 的章節表格已有「互動動畫」欄。章序維持 01 → 12。

## 每章網頁的教學流程

1. 保留原章節中文名稱、學習目標與原有小節順序。
2. 在對應觀念下放動畫舞台，圓點、方塊、線段、分界線與箭頭要真的移動或改變。
3. 提供「播放／暫停／重設／單步」，適合連續過程的場景另提供速度與時間軸。
4. 提供本概念的操作，例如調整 k、學習率、分類門檻、模型複雜度；操作後同步更新幾何圖形、數值與說明。
5. 在動畫旁提供短圖說、公式與原圖對照；不支援動畫時仍能閱讀核心解釋。
6. 頁面下方提供上一章、總目錄與下一章，順序固定為 01 → 12。

## 共用資源與技術選擇

共用資源集中於 [shared/](shared/README.md)。

- **Canvas 2D**：用在大量散點、連續路徑、梯度下降、KNN 距離與曲線變形。
- **原生 JavaScript**：管理時間軸與教學數值，不需後端、建置工具或外部套件。
- **Light Mode**：沿用既有教材的淺色視覺；類別除顏色外，也用形狀或標籤辨識。

網頁、樣式、程式、示範資料與所需圖片全部放在 `docs` 內。原教材圖片位於 `名詞解釋/`，不會隨 `/docs` 自動發布；網頁若使用原圖，需把使用到的圖片放入對應章節的 `assets/images/`。原文連結則指向 GitHub 中的來源檔案。

## 數值與動畫的界線

能直接計算的距離、資料切分、梯度、混淆矩陣與評估指標，讓幾何畫面反映實際計算。神經網路機制、搜尋策略或改善方案若只作概念動畫，明確標示「示意」，不把預設動畫結果寫成已訓練模型的效能。

教材特別處理下列易誤解的地方：

- 縮放器與特徵選擇只在訓練資料上擬合；分層抽樣的類別數量受整數樣本數限制。
- 損失曲線的收斂條件取決於函數與步長，不能把某個固定學習率當成通用最佳值。
- ROC-AUC 可能小於 0.5；指標分母為零時顯示未定義或明確約定，不產生假分數。
- R² 可以為負；標籤常數時的邊界情況另作說明。
- K 折驗證輪替前先保留最終測試集；Stacking 使用 OOF 預測來避免洩漏。
- 泛化鴻溝小不等於模型好，需一起看訓練／驗證誤差的絕對高度。
- `ceil(N / Batch Size)` 的計數預設保留最後不足一批的樣本；丟棄最後一批時另外說明。

## 更新與本機預覽

原教材位於 `名詞解釋/`。修改教材文字或 `chapters.json` 後，在專案根目錄執行：

```sh
.venv/bin/python docs/glossary/build.py
python3 -m http.server 8765 --directory docs
```

開啟 `http://localhost:8765/glossary/`。產生器使用 Mistune，將章節 Markdown、公式與表格轉成靜態 HTML，並複製引用圖片。網頁執行時只需瀏覽器，沒有 CDN 或後端依賴。

動畫修改於各章 `scenes/index.js`；共用播放器與計算工具在 `shared/`。動畫不需要重新產生 HTML，場景 ID 必須與 `chapters.json` 一致。數值檢查可執行 `node docs/glossary/check-math.mjs`。

沿用目前 GitHub Pages 的 `/docs` 設定。完成檔案 commit 並推送至 Pages 使用的分支後，等待 GitHub 部署完成。不要將發布來源改成 `docx`。
