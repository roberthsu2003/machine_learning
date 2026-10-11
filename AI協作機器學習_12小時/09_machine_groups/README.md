# 09｜設備保養：同一台機器不能偷看考卷

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。此資料夾目前是提示詞教材，尚未有模型程式或 Notebook。

- 任務：群組分割與分類。
- 最小應用：Gradio。
- 學生專案內位置：`09_machine_groups/`；共用專案根目錄的 `.venv`。
- 本例完成後：可用原始輸入執行預測，並說出一個模型限制。

## 資料與問題設定

本教材原創偽資料，data/machine_readings.csv，50 台假想設備各 20 次讀數，共 1,000 筆；seed=7009。

### 範例檔下載

- [下載 machine_readings.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/09_machine_groups/data/machine_readings.csv)：只下載本例 CSV，儲存到學生專案的 `09_machine_groups/data/machine_readings.csv`。
- [線上閱讀偽資料卡與欄位字典](./data/DATA_CARD.md)：直接在 GitHub 閱讀，不必下載。

若連結顯示文字，使用瀏覽器「另存新檔」，保留 `.csv` 副檔名。使用這份固定 CSV，不請 AI 重新生成；本例不需要其他範例的資料，也不必下載整個 repo。

- **X（特徵）**：temperature_c、vibration_mm_s、operating_hours、machine_age_years；machine_id、reading_id、is_synthetic 排除。
- **y（標籤）**：failure_next_7d（模擬未來七天是否故障）；沒有故障後才量到的欄位。
- **切分規則**：先用 GroupShuffleSplit(random_state=42) 留 20% 機器為 test；其餘機器再以 GroupShuffleSplit(random_state=43) 留 25% 為 val，約 30/10/10 台；三組 machine_id 不重疊。

## A｜一步一步建立 Notebook

先把本頁網址貼給 AI，再依序貼以下 prompt。
一次只貼一段，確認結果後再繼續；無法讀取網址時，貼上本頁內容。

### A0｜讓 AI 認識任務

```text
請閱讀我提供的教材與觀念連結，擔任我的機器學習助教。
沿用老師準備的環境，在 09_machine_groups/ 工作。
依本例設定逐步建立 lesson.ipynb，保留已完成內容。
用繁體中文說明，只做目前步驟；每步問我一個觀念問題。
能執行就直接執行；不能執行時明說，不編造成績。
Matplotlib／Seaborn 圖表的顯示文字全部使用繁體中文。
現在先解釋問題、特徵與預期結果，不開始訓練。
```

繪圖前，先貼[共用繁體中文繪圖 prompt](../README.md#共用-promptmatplotlib-與-seaborn-繁體中文繪圖)。
字型檔只需[下載一次](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/ChineseFont.ttf)，放在學生專案的 `fonts/ChineseFont.ttf`。

### A1｜讀取與觀察資料

觀念連結：[數據結構](../../名詞解釋/02_數據結構/README.md)

```text
在同一份 Notebook 載入本例資料。
讀取已下載的 data/machine_readings.csv，核對機器群組。
依上方設定確認 X／y，顯示前五筆、欄位與缺值。
套用共用字型，畫繁體中文圖，說明一列資料代表什麼。
缺檔時依上方下載連結取得，不換資料或重新生成。
```

### A2｜切分與前處理

觀念連結：[數據分割](../../名詞解釋/03_數據分割/README.md)、[模型泛化](../../名詞解釋/11_模型泛化/README.md)

```text
依上方「切分規則」分資料，不自行改成其他方式。
只用指定特徵，排除答案、ID 與生成提示欄位。
前處理只 fit train；交叉驗證時放在每折 Pipeline。
列出各份筆數，檢查重疊與洩漏，說明各自用途。
```

### A3｜訓練與比較模型

觀念連結：[常見算法](../../名詞解釋/07_常見算法/README.md)

```text
以 DummyClassifier 為基準，建立補值、縮放及邏輯迴歸。
使用 class_weight＝balanced，門檻由 val F1 選擇。
在開發集比較逐列與群組三折 CV，檢查機器是否重複。
比較只用 train／val 或開發集 CV，不用 test 挑設定。
執行後解釋比較結果；不要為了高分改資料。
```

### A4｜評估與解讀

觀念連結：[評估指標](../../名詞解釋/09_評估指標/README.md)、[模型泛化](../../名詞解釋/11_模型泛化/README.md)

```text
固定選好的模型與設定，再評估一次 test。
顯示 test precision、recall、F1 與混淆矩陣。
比較兩種 CV，解釋同機器重複造成的洩漏風險。
不要依 test 成績回頭調參；用生活語言說明結果。
```

### A5｜保存與確認結果

```text
重啟 kernel，從頭執行 lesson.ipynb，確認中文字型與圖表。
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
輸入四個感測／年齡欄位，顯示維護關懷分數。
沿用保存的門檻；不把 machine_id 當特徵。
載入已保存的完整模型，不在介面重新訓練。
以 app.py 的位置定位檔案，檢查輸入並標示來源與限制。
介面與圖表用繁體中文，圖表沿用共用字型設定。
僅供本機操作，提供啟動命令。
```

### B2｜驗收應用

```text
測試三組有效輸入，確認介面與 Notebook 預測一致。
檢查空白、非法輸入及缺少模型時的提示。
確認畫面與存檔圖表的繁體中文、負號均正常。
停止後重新啟動，確認仍能預測，修正實際發現的問題。
寫一份簡短 RUN.md，說明啟動、操作與模型限制。
回報已測試與待確認項目，再問我下方觀念問題。
```

## 觀念確認

1. 1000 列是否代表 1000 台不同機器？
2. 同一台機器出現在 train 和 test，會測到什麼？
3. 分數沒有明顯下降，就能說群組洩漏沒關係嗎？

## 完成檢查

三組機器 ID 交集均為空；row/group CV 僅在開發資料比較；app 的特徵不含 ID，報告清楚說明新機器評估範圍。

至少展示：一個真實執行的模型結果、一個介面輸入／輸出、一個限制；說明一項由你下命令、AI 幫你完成並經你檢查的工作。

[回到課程首頁](../README.md)
