# 06 優化算法：互動網頁規劃

原教材：[章節 README](../../../名詞解釋/06_優化算法/README.md)。
網頁檔案：`docs/glossary/06-optimization/index.html`。
發布網址：`https://roberthsu2003.github.io/machine_learning/glossary/06-optimization/`。

狀態：網頁與全部動畫場景已實作；推送至 GitHub 並完成 Pages 部署後，發布網址即可使用。

## 動畫與操作設計

| 場景 | 要看見的幾何動畫 | 學生／教師操作 |
| :--- | :--- | :--- |
| 梯度下降 | 小球沿 2D 損失曲線移動，切線與負梯度箭頭隨位置更新。 | 拖曳起點、單步下降、播放下降 |
| 學習率 | 三個小球並排顯示緩慢下降、穩定收斂與震盪／發散。 | 調整 α、比較三種步長 |
| Batch／SGD／Mini-batch | 樣本方塊分批進入模型，所選樣本的梯度箭頭加總，更新路徑隨批次大小改變。 | 切換方式、調整 Batch Size、固定隨機種子 |
| 優化器補充 | 在同一個教學損失地形上對照 SGD、Momentum 與 Adam 的運動路徑。 | 切換優化器、相同起點比較 |

## 原圖對應

動畫依原教材順序呈現；保留原圖對照，並在旁邊提供動畫的文字說明。
- `gradient-descent`：`01_gradient_descent.png`。
- `learning-rate`：`02_learning_rate.png`。
- `batch-size`：`03_batch_size_comparison.png`。
- `optimizers`：補充教學場景，沒有既有圖片。

## 檔案配置

```text
index.html            # 本章完整教材與場景入口
lesson.js             # 本章控制、數值計算與場景設定
scenes/index.js       # 本章所有場景的設定與數值繪圖
assets/               # 本章需隨 Pages 發布的原圖與資料
```

使用 `../shared/` 的共用樣式、動畫播放器、幾何與數值工具。所有執行資源都在 docs 內，沒有 CDN 或後端依賴。

## 驗收重點

- 能直接從本章網址進入，不依賴先打開總目錄。
- 動畫真的改變幾何位置、形狀、邊界或路徑，文字與數值同步解釋。
- 播放、暫停、重設與單步控制都能用；可重現相同資料與結果。
- 保留原教材的章節順序與中文名稱，導覽提供上一章／目錄／下一章。
- 計算結果與公式一致；純機制示意、預先設定數值與實際演算法清楚區分。
- Light Mode、手機版、鍵盤操作與無動畫文字說明皆可用。

[返回完整規劃](../README.md)
