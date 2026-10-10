# 11 模型泛化：互動網頁規劃

原教材：[章節 README](../../../名詞解釋/11_模型泛化/README.md)。
網頁檔案：`docs/glossary/11-generalization/index.html`。
發布網址：`https://roberthsu2003.github.io/machine_learning/glossary/11-generalization/`。

狀態：網頁與全部動畫場景已實作；推送至 GitHub 並完成 Pages 部署後，發布網址即可使用。

## 動畫與操作設計

| 場景 | 要看見的幾何動畫 | 學生／教師操作 |
| :--- | :--- | :--- |
| 未見資料與泛化 | 先用圓點呈現訓練樣本，再逐一投入菱形新樣本，觀察模型是否適用。 | 揭露新樣本、切換記憶型／規律型模型 |
| 泛化鴻溝 | 訓練與驗證誤差長條同步變動，以連接線顯示兩者差距與絕對高度。 | 切換欠擬合／過擬合／良好泛化 |
| 資料分割與 K 折交叉驗證 | 先封存測試方塊；剩餘資料分成 K 份，驗證份逐輪換位並累計分數。 | 設定 K、下一折、顯示平均與標準差 |
| 提升泛化 | 分別操作資料量、模型容量、正則化與訓練策略，對照新資料誤差。 | 選擇改善策略、查看假設與限制 |

## 原圖對應

動畫依原教材順序呈現；保留原圖對照，並在旁邊提供動畫的文字說明。
- `unseen-data`：`01_generalization_concept.png`。
- `generalization-gap`：`02_generalization_gap.png`。
- `cross-validation`：`03_data_splitting.png`, `04_cross_validation.png`。
- `improve-generalization`：`05_improve_generalization.png`。

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
