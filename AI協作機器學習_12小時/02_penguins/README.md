# 02｜企鵝：缺失值與類別欄位

你提出需求、看懂結果並驗收；AI 負責寫程式、執行和除錯。此資料夾目前是提示詞教材，尚未有模型程式或 Notebook。

- 任務：多類別分類。
- 最小應用：Gradio。
- 學生專案內位置：`02_penguins/`；共用專案根目錄的 `.venv`。
- 本例完成後：可用原始輸入執行預測，並說出一個模型限制。

## 一小時怎麼用

| 時間 | 學生工作 |
| :--- | :--- |
| 0～5 分 | 看下列觀念連結，貼 A0，先說明要解決的問題 |
| 5～15 分 | A1／A2：檢查資料、特徵與切分 |
| 15～35 分 | A3／A4：請 AI 訓練、做一個比較、解讀結果 |
| 35～40 分 | A5：從頭執行 Notebook，保存模型 |
| 40～55 分 | B1／B2：做最小介面與驗收 |
| 55～60 分 | 回答三個問題並展示結果 |

環境建立由老師帶領，這份教材從模型與資料的操作開始。若生成／下載延遲，先確保 Notebook 主流程與單筆應用完成，比較圖或批次功能可課後延伸。

## 觀念連結

- [數據結構](../../名詞解釋/02_數據結構/README.md)｜[線上原文](https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/02_數據結構/README.md)
- [數據分割](../../名詞解釋/03_數據分割/README.md)｜[線上原文](https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/03_數據分割/README.md)
- [特徵工程](../../名詞解釋/04_特徵工程/README.md)｜[線上原文](https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/04_特徵工程/README.md)
- [評估指標](../../名詞解釋/09_評估指標/README.md)｜[線上原文](https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/09_評估指標/README.md)

直接在 GitHub 開啟上述觀念連結閱讀，不必下載名詞解釋資料夾。AI 的步驟 prompt 已附可直接讀取的線上網址。

## 資料與問題設定

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)，344 筆、3 種企鵝；資料 CC0。資料由 Kristen Gorman 與 Palmer Station LTER 蒐集，整理套件由 Allison Horst 等人提供。

### 範例檔下載

[下載 penguins.csv](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv)，儲存到學生專案的 `02_penguins/data/penguins.csv`。若瀏覽器顯示文字，使用「另存新檔」，保留 `.csv` 副檔名。只需要這一份資料；也可讓 A1 的 AI 從同一網址下載。

- **X（特徵）**：bill_length_mm、bill_depth_mm、flipper_length_mm、body_mass_g 為數值；island、sex 為無序類別；year 不納入本例。
- **y（標籤）**：species；只有缺 target 的列可以刪除，特徵缺失要由 Pipeline 處理。
- **切分規則**：stratified 80/20 開發／test，random_state=42；開發資料 3 折分層 CV。先切分，不能先對全部資料補值或編碼。

## A｜一步一步請 AI 建立 ipynb

老師帶領完成環境後，直接從 GitHub 複製 A0～A5，依序貼給 AI。一次只貼一段；每段執行成功、回答 AI 的觀念問題後才貼下一段。另開 AI 對話時，重新貼 A0 並請它讀現有檔案接續。

### A0｜設定 AI 的工作方式

```text
你是我的機器學習程式執行者與教學助教，我不熟悉 Python。
老師已帶領我準備好學生專案與環境；請沿用目前開啟的學生專案，不重新建立環境。
本例資料夾是 02_penguins；不存在時請在學生專案內建立。教材不一定在本機，請讀以下 README 網址或我貼上的教材內容：
https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/02_penguins/README.md
本例是「企鵝：缺失值與類別欄位」，任務是多類別分類，之後介面使用 Gradio。
你負責產生／修改真正可開啟的 02_penguins/lesson.ipynb，不要只貼 Python 程式碼讓我自己建立檔案。
每個步驟在同一 Notebook 追加 Markdown 解說與少量程式 cell，不覆寫我已完成的觀察。
程式與圖用繁體中文說明；用語言比喻先解釋，再寫程式。繪圖中文若缺字型，選可用字型或英文圖例＋中文 Markdown，不讓缺字阻擋課程。
只執行我目前貼出的步驟；不要一次完成 Notebook、app 和全部延伸。
能存取檔案／終端時請直接做，否則提供完整檔案附件與一條可複製命令，說明我必須在哪個資料夾執行。
沿用老師準備的環境；缺少套件時列出名稱交由老師協助，不重建或更換環境。
Notebook kernel 必須是根目錄 .venv；每個新函式讓初學者看懂用途，不要求我手寫程式。
只在本例新增資料、Notebook、artifacts、app 檔案，不改其他例、不覆寫教材 README 或固定偽資料。
先讀以下連結中最相關的 1～2 章段落，不必全章全讀，摘取本例的 2～3 個觀念；無法讀取時明講，讓我開連結提供段落，不捏造老師教材：
https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/02_數據結構/README.md
https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/03_數據分割/README.md
https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/04_特徵工程/README.md
https://github.com/roberthsu2003/machine_learning/blob/main/名詞解釋/09_評估指標/README.md
現在只說明問題、X／y（非監督任務說明沒有 y）、預期輸入／輸出與工作步驟；問我一個能用自然語言回答的問題，然後停下。
```

### A1｜載入與看懂資料

```text
承接 A0，只做資料載入與觀察。
建立／追加 02_penguins/lesson.ipynb 的「問題與資料」區塊。
優先讀取我下載的 data/penguins.csv；若不存在，從 https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv 下載並快取同一路徑，保留原始空值；source.md 記錄下載時間、SHA-256 與原作者。下載失敗就停在本步，使用教師預先快取的同檔，不能偷偷換偽資料。
資料與 artifacts 路徑要在該例資料夾內。Notebook 可能由專案根目錄或本例資料夾啟動，先檢查工作目錄與存在檔案，正確定位，不寫死教師電腦路徑。
請執行載入，顯示前 5 筆、列／欄數、欄名、型態、缺失與一張最有用的圖；有 target 才列類別／target 分布，分群則列六欄消費分布。
這次不訓練模型。資料下載失敗不換資料、不編造讀取結果，先說明可重試或使用教師同來源快取的辦法。
來源、授權、原始／子集／偽資料身分要寫入 Markdown；下載的開放資料另存來源 URL、日期、檔案 SHA-256 與引用。
X：bill_length_mm、bill_depth_mm、flipper_length_mm、body_mass_g 為數值；island、sex 為無序類別；year 不納入本例。
y：species；只有缺 target 的列可以刪除，特徵缺失要由 Pipeline 處理。
用一列說明資料如何變成模型輸入，問我哪個欄位不能放進 X，等我回答。
```

### A2｜切分與預處理

```text
承接 A1，只做切分及前處理設計。
規則：stratified 80/20 開發／test，random_state=42；開發資料 3 折分層 CV。先切分，不能先對全部資料補值或編碼。
依明確白名單建立 X，排除 target、ID、稽核／生成提示欄位。
先切分再 fit 補值、縮放、編碼／文字特徵；若使用交叉驗證，前處理必須放在每折的 Pipeline 裡。
請在 Notebook 畫一張 train／val／test 或 CV 的角色圖，列出樣本數並用可執行檢查證明索引、群組或日期符合本例規則。
分群則按照本例探索分析規則，不硬加不存在的 y 或分類 accuracy。
現在不看 test 成績選參數。先解釋本例為何使用這種切分，問我「哪一份資料用來調整模型？」後停下。
```

### A3｜建立可比較的模型

```text
承接 A2，現在才在同一 Notebook 訓練模型。
本例實作範圍：DummyClassifier 基準；ColumnTransformer：數值 SimpleImputer(median)+StandardScaler；類別 SimpleImputer(most_frequent)+OneHotEncoder(handle_unknown="ignore")；後接 LogisticRegression(max_iter=2000)。不要把 island／sex 映成有大小的 0、1、2。
每次使用固定 random_state=42（群組第二次分割為 43）；保持相同切分與評估規則。
不要自行升級成深度學習、雲端服務或大量調參；不要為了高分刪除不利測試列或更改標籤。
追加一個最小比較實驗：用開發資料比較「只用數值」與「數值＋類別」的 3 折 macro-F1。解釋類別編碼增加了哪些欄，不宣稱加入欄位一定變好。
比較只在 train／val／開發 CV 上做；結果與圖必須由實際執行產生，不能填預設成績。
說清楚哪些是模型學到的參數、哪些是我們選的超參數，並保存此次比較的設定與原始數值。
問我觀察到什麼差異；如果結果與預期不同，說明此次證據的限制，不改結果迎合教學。
```

### A4｜評估與解讀

```text
承接 A3，先固定已選模型及設定，再做本例的最終評估。
評估規格：test accuracy、macro-F1、混淆矩陣；顯示每欄缺失數、編碼後特徵名；拿一列缺 sex 的輸入驗證補值流程。
不要用最後 test 成績再調模型、門檻或選特徵；若此例是分群，忠實區分探索指標與未見客戶分派。
把指標、圖表與資料切分記錄寫進 Notebook，並用一句生活語言解釋「這個數值代表什麼」。
示範一個錯誤／限制或有差距的案例；不能只展示成功樣本。
拿與本例相關的原教材連結提示我重新看哪個觀念。
若模型沒有超過基準，照實說明，不宣稱完成了優於基準的模型。
問我這個模型適合什麼輸入、不適合什麼情境，等我回答。
```

### A5｜從頭驗收與保存模型

```text
承接 A4，現在完成 Notebook 主流程，不先建立 UI。
重啟 kernel 後從第一格到最後一格執行 02_penguins/lesson.ipynb；修正任何隱藏執行順序依賴。
保存真正可用的完整前處理＋模型到 02_penguins/artifacts/model.joblib。
一般模型保存 Pipeline；若為 SGD 教學，保存 X scaler、模型、y scaler 的完整 bundle；分群保存 log／縮放／KMeans；文字保存 TF-IDF／NB，不能只保存最後 estimator。
另存 artifacts/metadata.json：任務、特徵名稱／順序、單位、類別映射、資料來源／雜湊、切分、random_state、模型／超參數、已執行指標、套件版本、model_version、建立日期及適用限制；需要的門檻、基準值、日期或繪圖資料也保存。
本例特殊驗收：數值／類別欄位清單與 target 分離；缺欄、空值、未知類別都有清楚處理；app 與 Notebook 同一原始輸入預測相同。
另外用新 Python 程序載入 artifacts，在同一原始輸入上比對 Notebook 預測（浮點用合理容差），確認不是依賴 Notebook 記憶體裡的變數。
請報告哪些檢查真正執行、結果與檔案位置；若你無法執行，標示「待學生執行」，不可宣稱通過。
用 100 字內讓我能口頭說出需求、資料、模型、評估與限制，然後停下。
```

## B｜學完 ipynb 後，請 AI 做成真正可操作的專案

先通過 A5，再依序貼 B1／B2。核心目標是本機、單頁、可重啟的最小應用；雲端部署、登入、資料庫與大型視覺效果不列入這 1 小時。需要桌面程式時可另用[課程首頁的 PyQt6 延伸 prompt](../README.md#想改成-pyqt6-桌面程式時)。

### B1｜以已完成模型建立應用

```text
延續 A0～A5；若是新對話，先讀 A0 提供的線上教材（或我貼上的內容），以及本機 02_penguins/lesson.ipynb 與 02_penguins/artifacts/metadata.json。
用 Gradio 在 02_penguins/app.py 建立繁體中文本機應用，必要的共同推論函式放在同例 inference.py。
先確認 model.joblib 與 metadata 存在；若沒有或版本不相容，提示回到 A5，不在 app 啟動時自動重訓或下載資料。
使用老師準備好的 gradio；查已安裝版本的官方 API，缺少套件時列出需求交由老師協助。
功能：四個有單位的數字欄、island／sex 下拉選單及「未提供」選項；未提供要轉成缺失值交給既有 Pipeline，顯示企鵝種類 species 和三類估計機率。
載入一次完整模型；輸入先驗證欄位、順序、型態與單位，保持 Notebook 的前處理，不另寫一套手動轉換。
所有 artifacts 與資料檔都依 app.py 的檔案位置定位，不依賴目前終端工作目錄。
頁面要顯示例子名稱、任務、來源／偽資料身分、已存的評估結果與限制；不可將模型分數稱為真實保證。
Gradio 使用本機網址與 share=False；Streamlit 只建立本機單頁。這次不做 FastAPI、雲端、登入、付款或資料庫。
提供從學生專案根目錄執行的一條命令：uv run 02_penguins/app.py
實際啟動並試一筆輸入；若沒有瀏覽器操作能力，提供我可照著按的 3 步，而不是宣稱已驗證介面。
```

### B2｜驗收與學生說明

```text
只做 02_penguins 應用驗收與修正，不新增大型功能。
檢查：1. 三組有效原始輸入的 app 預測與同一 artifacts 載入的推論一致。
2. 本例合理的空白、缺欄、型態錯誤或非法數值會得到清楚提示；允許缺值的欄位仍交給同一 Pipeline 補值。
3. 停止後從學生專案根目錄以「uv run 02_penguins/app.py」重啟仍能預測，不重新訓練、不讀取 Notebook 的全域變數。
4. 沒有 artifacts 時說明要先完成 A5；保存的評估圖／指標不是硬寫成績。
5. 本例特殊要求仍成立：數值／類別欄位清單與 target 分離；缺欄、空值、未知類別都有清楚處理；app 與 Notebook 同一原始輸入預測相同。
只修真正發現的問題，報告實際執行與尚待操作的驗收項目。
最後更新本例另建的 RUN.md（保留教材 README），說明輸入／輸出、操作、模型來源、評估及限制，並給我三分鐘展示講稿。
結束前逐題問我本例的三個觀念問題，等我用自然語言回答；依原教材概念糾正，不要求我背程式語法。
```

## 學生下課前要能回答

1. 性別沒填可以一律當成 0 嗎？
2. 為什麼補值的中位數只用訓練資料算？
3. One-Hot 的 1 是大小還是屬於該類？

## 今天的驗收

數值／類別欄位清單與 target 分離；缺欄、空值、未知類別都有清楚處理；app 與 Notebook 同一原始輸入預測相同。

至少展示：一個真實執行的模型結果、一個介面輸入／輸出、一個限制；說明一項由你下命令、AI 幫你完成並經你檢查的工作。

[回到 12 小時課程](../README.md)
