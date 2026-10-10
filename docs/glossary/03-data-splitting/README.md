# 03 數據分割：互動網頁規劃

原教材：[章節 README](../../../名詞解釋/03_數據分割/README.md)。
網頁檔案：`docs/glossary/03-data-splitting/index.html`。
發布網址：`https://roberthsu2003.github.io/machine_learning/glossary/03-data-splitting/`。

狀態：網頁與全部動畫場景已實作；推送至 GitHub 並完成 Pages 部署後，發布網址即可使用。

## 動畫與操作設計

| 場景 | 要看見的幾何動畫 | 學生／教師操作 |
| :--- | :--- | :--- |
| 訓練／測試分割 | 帶 ID 的樣本方塊打散，再移入互不重疊的訓練與測試容器。 | 調整比例、固定種子重新分割 |
| 三份資料的角色 | 資料移入 Train／Validation／Test；只有 Train 流入模型，Validation 流向調參旋鈕，Test 保持封存。 | 切換訓練／調參／最終評估、調整比例 |
| 隨機、分層與時間切分 | 切換抽樣方式後，兩類樣本在各容器的比例改變；時間模式按先後排列方塊。 | 切換切分方式、顯示各類數量 |
| 資料洩漏與診斷 | 對照「全資料 fit」與「只用 Train fit」；測試資訊以橘色路徑表示是否流入預處理。 | 正確／錯誤流程對照、顯示訓練與驗證誤差 |

## 原圖對應

動畫依原教材順序呈現；保留原圖對照，並在旁邊提供動畫的文字說明。
- `train-test`：`01_train_test_split.png`。
- `train-validation-test`：`02_three_way_split_validation.png`。
- `split-strategies`：`02_three_way_split_validation.png`。
- `leakage`：`03_split_workflow_and_overfitting.png`。

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
