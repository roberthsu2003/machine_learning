# 02｜企鵝：缺失值與類別欄位

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。此資料夾目前是提示詞教材，尚未有模型程式或 Notebook。

- 任務：多類別分類。
- 最小應用：Gradio。
- 學生專案內位置：`02_penguins/`；共用專案根目錄的 `.venv`。
- 本例完成後：可用原始輸入執行預測，並說出一個模型限制。

## 資料與問題設定

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)，344 筆、3 種企鵝；資料 CC0。資料由 Kristen Gorman 與 Palmer Station LTER 蒐集，整理套件由 Allison Horst 等人提供。

### 範例檔下載

[下載 penguins.csv](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv)，儲存到學生專案的 `02_penguins/data/penguins.csv`。若瀏覽器顯示文字，使用「另存新檔」，保留 `.csv` 副檔名。只需要這一份資料；也可讓 A1 的 AI 從同一網址下載。

- **X（特徵）**：bill_length_mm、bill_depth_mm、flipper_length_mm、body_mass_g 為數值；island、sex 為無序類別；year 不納入本例。
- **y（標籤）**：species；只有缺 target 的列可以刪除，特徵缺失要由 Pipeline 處理。
- **切分規則**：stratified 80/20 開發／test，random_state=42；開發資料 3 折分層 CV。先切分，不能先對全部資料補值或編碼。

## 觀念連結

- [數據結構](../../名詞解釋/02_數據結構/README.md)
- [數據分割](../../名詞解釋/03_數據分割/README.md)
- [特徵工程](../../名詞解釋/04_特徵工程/README.md)
- [評估指標](../../名詞解釋/09_評估指標/README.md)

## A｜一步一步建立 Notebook

先把本頁網址貼給 AI，再依序貼以下 prompt。
一次只貼一段，確認結果後再繼續；無法讀取網址時，貼上本頁內容。

### A0｜讓 AI 認識任務

```text
請閱讀我提供的教材與觀念連結，擔任我的機器學習助教。
沿用老師準備的環境，在 02_penguins/ 工作。
依本例設定逐步建立 lesson.ipynb，保留已完成內容。
用繁體中文說明，只做目前步驟；每步問我一個觀念問題。
能執行就直接執行；不能執行時明說，不編造成績。
現在先解釋問題、特徵與預期結果，不開始訓練。
```

### A1｜讀取與觀察資料

```text
在同一份 Notebook 載入本例資料。
優先讀取 data/penguins.csv；缺檔時從上方連結下載。
依上方設定確認 X／y，顯示前五筆、欄位與缺值。
畫一張有用的圖，說明一列資料代表什麼。
缺檔時依上方下載連結取得，不換資料或重新生成。
```

### A2｜切分與前處理

```text
依上方「切分規則」分資料，不自行改成其他方式。
只用指定特徵，排除答案、ID 與生成提示欄位。
前處理只 fit train；交叉驗證時放在每折 Pipeline。
列出各份筆數，檢查重疊與洩漏，說明各自用途。
```

### A3｜訓練與比較模型

```text
以 DummyClassifier 為基準，建立邏輯迴歸 Pipeline。
數值用中位數補值並縮放；類別補值後做 One-Hot。
比較一筆缺少 sex 的輸入，解釋如何處理。
比較只用 train／val 或開發集 CV，不用 test 挑設定。
執行後解釋比較結果；不要為了高分改資料。
```

### A4｜評估與解讀

```text
固定選好的模型與設定，再評估一次 test。
顯示 test accuracy、macro-F1 與混淆矩陣。
列出缺值數與編碼後欄位，說明一個限制。
不要依 test 成績回頭調參；用生活語言說明結果。
```

### A5｜保存與確認結果

```text
重啟 kernel，從頭執行 lesson.ipynb，修正錯誤。
將完整前處理與模型保存到 artifacts/model.joblib。
另存 metadata.json：欄位、來源、切分、設定、指標與版本。
本例需要的門檻、scaler、圖表與範例也一起保存。
用新程序載入，比對同一輸入的預測，回報實際結果。
```

## B｜把模型做成應用

完成 A5 後再貼 B1、B2。桌面版可使用[PyQt6 延伸 prompt](../README.md#想改成-pyqt6-桌面程式時)。

### B1｜建立介面

```text
讀取本例 Notebook 與 artifacts，用 Gradio 建立 app.py。
輸入四個數值及 island／sex，顯示企鵝種類與估計機率。
允許未提供的欄位，交由保存的 Pipeline 處理。
載入已保存的完整模型，不在介面重新訓練。
以 app.py 的位置定位檔案，檢查輸入並標示來源與限制。
介面用繁體中文、僅供本機操作，提供啟動命令。
```

### B2｜驗收應用

```text
測試三組有效輸入，確認介面與 Notebook 預測一致。
檢查空白、非法輸入及缺少模型時的提示。
停止後重新啟動，確認仍能預測，修正實際發現的問題。
寫一份簡短 RUN.md，說明啟動、操作與模型限制。
回報已測試與待確認項目，再問我下方觀念問題。
```

## 觀念確認

1. 性別沒填可以一律當成 0 嗎？
2. 為什麼補值的中位數只用訓練資料算？
3. One-Hot 的 1 是大小還是屬於該類？

## 完成檢查

數值／類別欄位清單與 target 分離；缺欄、空值、未知類別都有清楚處理；app 與 Notebook 同一原始輸入預測相同。

至少展示：一個真實執行的模型結果、一個介面輸入／輸出、一個限制；說明一項由你下命令、AI 幫你完成並經你檢查的工作。

[回到課程首頁](../README.md)
