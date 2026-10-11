# 12 機器學習訓練過程：互動網頁規劃

原教材：[章節 README](../../../名詞解釋/12_機器學習訓練過程/README.md)。
網頁檔案：`docs/glossary/12-training-process/index.html`。
發布網址：`https://roberthsu2003.github.io/machine_learning/glossary/12-training-process/`。

狀態：網頁與全部動畫場景已實作；推送至 GitHub 並完成 Pages 部署後，發布網址即可使用。

## 動畫與操作設計

| 場景 | 要看見的幾何動畫 | 學生／教師操作 |
| :--- | :--- | :--- |
| 端到端生命週期 | 六階段依蛇形箭頭前進，每格顯示產出；部署監控以橘色閉環回到問題與資料。 | 逐階段播放、滑桿選擇階段、查看產出與重訓回饋 |
| 訓練迭代與 Epoch／Batch／Iteration | 一批走完前向、損失、反傳與更新才完成一次 Iteration；單步、批次 ID 與 Epoch 計數一致。 | 設定 N／Batch Size／Epoch、單步完成一批、切換 drop_last |
| 健康度與檢查清單 | 誤差長條以共用 0–1 尺度對照；三個流程關卡各有文字標籤與通過狀態。 | 切換擬合狀態、檢查資料切分／fit Train／Test 不調參 |

## 原圖對應

動畫依原教材順序呈現；保留原圖對照，並在旁邊提供動畫的文字說明。
- `lifecycle`：`01_ml_lifecycle_pipeline.png`。
- `inner-loop`：`02_model_training_iteration_loop.png`。
- `training-health`：`03_underfitting_overfitting_spectrum.png`。

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
