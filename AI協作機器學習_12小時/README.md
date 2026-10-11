![AI Python 機器學習實作入門](./course-cover.png)

# AI 協作機器學習｜十二個範例

這條課程路線為不熟悉程式的學生設計。學生以自然語言提出需求，AI 建立並執行程式；學生負責看懂資料、選擇評估方式、解讀結果與驗收。每例先用 ipynb 學概念，再把同一模型做成 Gradio 或 Streamlit 本機應用。

教材目前提供 **12 份範例 README、逐步 prompt，以及後六例的固定偽資料**。沒有預先代寫 `.py`、`.ipynb`、模型或學生專案的環境檔；這些成果要由學生下命令，讓 AI 在學生自己的專案產生。

## 十二個範例

| 編號 | 範例／資料夾 | 資料 | 核心觀念 | 完成的應用 |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [鳶尾花分類](./01_iris/README.md) | 開放：Iris | X／y、分類、分層切分、KNN、基準 | Gradio 花種預測 |
| 02 | [企鵝種類](./02_penguins/README.md) | 開放：Palmer Penguins | 缺失、One-Hot、Pipeline、縮放 | Gradio 企鵝分類 |
| 03 | [汽車油耗](./03_auto_mpg/README.md) | 開放：Auto MPG | 回歸、MAE／RMSE／R²、類別特徵 | Streamlit 油耗預測 |
| 04 | [手寫數字](./04_digits/README.md) | 開放：Digits 子集 | 圖片特徵、分類標籤、混淆矩陣 | Gradio 8×8 測試圖辨識 |
| 05 | [葡萄酒分類](./05_wine/README.md) | 開放：Wine | 超參數、樹深、森林、CV、過度擬合 | Streamlit 品種分類 |
| 06 | [批發客戶分群](./06_wholesale/README.md) | 開放：Wholesale Customers | 非監督、尺度、群輪廓、投影限制 | Streamlit 客戶群輪廓 |
| 07 | [福利社營業額](./07_shop_sales/README.md) | 偽資料：600 筆 | 權重、學習率、epoch、雜訊 | Gradio 營業額與訓練曲線 |
| 08 | [訂閱服務流失](./08_customer_churn/README.md) | 偽資料：1,000 筆 | 不平衡、precision／recall、門檻 | Streamlit 關懷候選清單 |
| 09 | [設備維護](./09_machine_groups/README.md) | 偽資料：50 台×20 筆 | 群組分割、洩漏、新設備泛化 | Gradio 維護關懷示範 |
| 10 | [咖啡店每日銷量](./10_cafe_time/README.md) | 偽資料：730 天 | 時間切分、lag、只能用過去 | Streamlit 次日銷量預估 |
| 11 | [客服工單](./11_support_text/README.md) | 偽資料：900 則、30 模板群 | TF-IDF、NB、模板洩漏 | Gradio 工單分流 |
| 12 | [用電與漂移](./12_energy_drift/README.md) | 偽資料：600 天×3 棟 | 完整流程、未來泛化、誤差監測 | Streamlit 預測與批次監測 |

## 在 GitHub 閱讀，只下載當次需要的檔案

環境建立由老師課堂示範，本教材從已準備好的學生專案開始。

1. 在 GitHub 開啟上方當次範例的 README，直接複製 prompt 給 AI，不必下載 README。
2. 在老師帶領建立的學生專案中，讓 AI 建立該例資料夾，例如 `01_iris/`；Notebook、模型與應用都放在該例內。
3. 本例有提供資料檔時，點該例「範例檔下載」連結，只儲存本次需要的檔案。不要下載整個 GitHub repo 或全部十二例資料。
4. 依序貼 A0～A5，再貼 B1～B2。觀念解說直接開 GitHub 的「名詞解釋」連結閱讀。

CSV 連結開啟原始檔；若瀏覽器顯示文字，使用「另存新檔」，保留 `.csv` 副檔名，放到表格指定的位置。資料卡可線上閱讀，不必下載。AI 可讀取教材網址；如果工具無法讀取，貼上當次 README 的內容即可，不必把教材複製到本機。

### 後六例的範例檔下載

| 範例 | 只下載這個 CSV | 放在學生專案內 | 線上欄位說明 |
| :--- | :--- | :--- | :--- |
| 07 | [下載 shop_sales.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/07_shop_sales/data/shop_sales.csv) | `07_shop_sales/data/shop_sales.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/07_shop_sales/data/DATA_CARD.md) |
| 08 | [下載 customer_churn.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/08_customer_churn/data/customer_churn.csv) | `08_customer_churn/data/customer_churn.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/08_customer_churn/data/DATA_CARD.md) |
| 09 | [下載 machine_readings.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/09_machine_groups/data/machine_readings.csv) | `09_machine_groups/data/machine_readings.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/09_machine_groups/data/DATA_CARD.md) |
| 10 | [下載 cafe_daily.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/10_cafe_time/data/cafe_daily.csv) | `10_cafe_time/data/cafe_daily.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/10_cafe_time/data/DATA_CARD.md) |
| 11 | [下載 support_tickets.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/11_support_text/data/support_tickets.csv) | `11_support_text/data/support_tickets.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/11_support_text/data/DATA_CARD.md) |
| 12 | [下載 energy_daily.csv](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/12_energy_drift/data/energy_daily.csv) | `12_energy_drift/data/energy_daily.csv` | [資料卡](https://github.com/roberthsu2003/machine_learning/blob/main/AI%E5%8D%94%E4%BD%9C%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_12%E5%B0%8F%E6%99%82/12_energy_drift/data/DATA_CARD.md) |

前六例使用開放資料：01、04、05 由 AI 載入 scikit-learn 內附資料；02 的 README 提供企鵝 CSV 下載連結；03、06 由 A1 從 UCI 載入並快取本例資料。每次只取得當前例子的資料。

## 每例的兩組 prompt 怎麼貼

- **A0～A5：Notebook 路線。** 設定 AI 角色 → 看資料 → 切分與前處理 → 建模／比較 → 評估 → 重新執行與保存。每段停下問一個概念問題，學生確認成功後才貼下一段。
- **B1～B2：專案路線。** 讀取同一已保存模型 → 建立介面 → 驗收重啟與預測一致性。UI 不重新訓練，避免介面與 Notebook 變成兩個不同模型。

AI 要能直接建立真正的 `.ipynb` 與程式檔。若使用只有聊天功能的 AI，請它交付完整檔案附件，再由學生存到指定路徑；沒有工具能力時不可宣稱已執行或測試。每次續接對話讓 AI 讀線上教材，以及本機 Notebook 與 metadata，避免從頭亂改。

### 共用 prompt｜Matplotlib 與 Seaborn 繁體中文繪圖

第一次繪圖前貼給 AI；每個新 Notebook 都要套用，應用程式也沿用同一設定。

[下載 ChineseFont.ttf](https://raw.githubusercontent.com/roberthsu2003/machine_learning/main/ChineseFont.ttf)，放在學生專案的 `fonts/ChineseFont.ttf`，十二例共用，不必下載整個 repo。若瀏覽器開啟檔案，請另存並保留 `.ttf` 副檔名。

參考你的[邏輯迴歸範例](../邏輯迴歸/03_diabetes_logistic_regression.ipynb)的字型註冊方式，以及[混淆矩陣範例](../機器學習常使用的圖表/混淆矩陣confusion_maxtrix.ipynb)的中文標題與標籤。資料欄位保留原名，圖表顯示名稱改成繁體中文。

```text
請在 Notebook 第一張圖之前建立共用字型設定。
定位 fonts/ChineseFont.ttf，用 fontManager.addfont 註冊。
用 FontProperties(fname=字型路徑).get_name() 取得字型名稱。
Matplotlib 與 sns.set_theme(font=字型名稱) 使用相同字型。
設定 axes.unicode_minus=False；重啟或應用啟動時重新註冊。
圖表標題、座標軸、圖例、類別與註解全部使用繁體中文。
用含負數的中文圖測試顯示及存檔；缺字時修正，不改成英文。
```

缺少字型檔時，請 AI 提供上方下載連結；不要寫死老師電腦的路徑。切換 Seaborn 主題時仍保留中文字型。設定依據：[Matplotlib 字型管理](https://matplotlib.org/stable/api/font_manager_api.html)、[Seaborn 主題設定](https://seaborn.pydata.org/generated/seaborn.set_theme.html)。

### AI 出錯時的除錯 prompt

```text
這是我執行目前步驟時的錯誤：
（貼上完整錯誤與步驟）
請說明原因，只修正必要部分並重跑。
保留原資料、切分與已完成內容，回報結果與下一步。
```

### 想改成 PyQt6 桌面程式時

這是可替代 B1 的延伸；選一個已完成案例即可。官方資料：[PyQt6](https://www.riverbankcomputing.com/static/Docs/PyQt6/)。

```text
請將我指定範例的已保存模型，做成 PyQt6 本機桌面應用。
沿用原始輸入、前處理與模型，不重新訓練。
建立 desktop_app.py，提供輸入、預測按鈕與結果。
Matplotlib／Seaborn 圖表沿用共用字型，全部顯示繁體中文。
保留原本的網頁應用，提供啟動命令並依 B2 驗收。
不能操作視窗時，列出需要我確認的項目。
```

主要應用依據 [Gradio 入門](https://www.gradio.app/guides/quickstart) 與 [Streamlit 基本概念](https://docs.streamlit.io/get-started/fundamentals/main-concepts)。教學以能操作、能驗收的本機專案為成果。

## 名詞解釋如何穿插教學

各段 prompt 的標題下方，依需要連到本倉庫[名詞解釋](../名詞解釋/README.md)，學生直接在 GitHub 閱讀。先開與當前問題相關的動畫／段落，再讓 AI 以實際結果回扣觀念。

| 原教材章節 | 最適合回看的範例 |
| :--- | :--- |
| [01 機器學習的類型](../名詞解釋/01_機器學習的類型/README.md) | 01、03、04、06：分類、回歸、無監督 |
| [02 數據結構](../名詞解釋/02_數據結構/README.md) | 01、02、04、11：X／y、圖片、文字 |
| [03 數據分割](../名詞解釋/03_數據分割/README.md) | 01、08、09、10：分層、群組、時間 |
| [04 特徵工程](../名詞解釋/04_特徵工程/README.md) | 02、03、06、11：缺值、編碼、縮放、文字 |
| [05 模型參數](../名詞解釋/05_模型參數/README.md) | 05、07：樹深與權重 |
| [06 優化算法](../名詞解釋/06_優化算法/README.md) | 07：真實 epoch 與學習率 |
| [07 常見算法](../名詞解釋/07_常見算法/README.md) | 01、05、11：KNN、樹、NB |
| [08 集成學習](../名詞解釋/08_集成學習/README.md) | 05、12：森林、Bagging |
| [09 評估指標](../名詞解釋/09_評估指標/README.md) | 01、03、08、12：分類與回歸評估 |
| [10 模型性能問題](../名詞解釋/10_模型性能問題/README.md) | 05、07、12：泛化差距與擬合 |
| [11 模型泛化](../名詞解釋/11_模型泛化/README.md) | 05、09、10、12：新樣本、新設備、未來 |
| [12 機器學習訓練過程](../名詞解釋/12_機器學習訓練過程/README.md) | 07、10、12：訓練、保存、推論與監測 |

[名詞解釋互動網頁](https://roberthsu2003.github.io/machine_learning/glossary/) 可在課堂直接播放。

## 後六例的偽資料安排

後六例共提供 **6,030 筆** CSV，均為完全合成資料。每份資料卡記錄欄位角色／單位、筆數、種子、生成規則、實際類別筆數與 SHA-256。

- 07：帶雜訊的線性營業額，便於看學習率與訓練曲線。
- 08：少數流失類別與缺值，便於理解基準、門檻及漏報／誤報。
- 09：設備內重複觀測，便於理解群組分割的目的。
- 10：連續日期、事前資訊與事後銷量，便於驗證只用歷史特徵。
- 11：模板內重複與少量標籤雜訊，便於理解文字特徵和模板洩漏。
- 12：後期規則改變，便於理解模型保存、未來評估與監測。

資料生成規則由教師設計；特徵白名單排除標籤、ID、生成狀態與 `is_synthetic`。AI 不因結果不漂亮而重抽資料。假想情境的模型表現不能直接視為真實商業效益。

## 教師課前準備與最後驗收

課前在學生使用的 AI 工具與 Python 3.12 環境試跑 prompt；02、03、06 需外部下載，教師可先快取同來源檔案與來源記錄。教材尚未執行學生將生成的十二個模型／介面，來源載入和框架 API 由 AI 依安裝版本確認，不能把這份教材的檔案檢查當成模型測試通過。

完成範例後請回答：「我的 X／y 是什麼？我怎麼分資料？我用什麼證據相信結果？AI 幫我執行了什麼，我如何確認？」

最後每位學生選一個完成案例，展示需求、資料來源、切分、基準與模型、介面輸入／輸出及一個限制。評量重點為觀念與驗收，不考手寫 Python 語法。完成本課程代表能指揮 AI 建立、執行與檢查初階機器學習專案；後續可使用原教材深化算法與工程。

[返回主教材](../README.md)
