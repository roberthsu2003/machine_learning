# 09 評估指標：互動網頁規劃

原教材：[章節 README](../../../名詞解釋/09_評估指標/README.md)。
網頁檔案：`docs/glossary/09-evaluation-metrics/index.html`。
發布網址：`https://roberthsu2003.github.io/machine_learning/glossary/09-evaluation-metrics/`。

狀態：網頁與全部動畫場景已實作；推送至 GitHub 並完成 Pages 部署後，發布網址即可使用。

## 動畫與操作設計

| 場景 | 要看見的幾何動畫 | 學生／教師操作 |
| :--- | :--- | :--- |
| 混淆矩陣 | 每個樣本按真實類別與預測類別移進 TP／FP／FN／TN 四格，數量同步增加。 | 逐樣本分類、調整分類門檻 |
| 分類指標與 ROC | 改變門檻後，圓點重新落格，Accuracy／Precision／Recall／F1 及 ROC 游標同步更新。 | 門檻滑桿、正類比例、查看指標分母 |
| MAE、MSE、RMSE | 殘差線段展開為絕對長度與平方色塊，加入離群點後比較各種誤差變化。 | 移動預測線、加入／移除離群值 |
| R² | 同時顯示平均值基準線與模型線，對照 SST／SSE 平方塊大小。 | 切換基準／模型、嘗試負 R² |

## 原圖對應

動畫依原教材順序呈現；保留原圖對照，並在旁邊提供動畫的文字說明。
- `confusion-matrix`：`01_confusion_matrix.png`。
- `classification-metrics`：`02_classification_metrics.png`。
- `regression-errors`：`03_regression_metrics_errors.png`。
- `r-squared`：`04_r_squared.png`。

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
