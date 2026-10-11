# 名詞解釋網頁動畫檢查紀錄

檢查日期：2026-10-11。範圍：12 章、42 個互動場景。

全部互動場景使用 Canvas 2D。原教材 SVG 為靜態對照圖片，並非動畫引擎。這批教學內容以二維資料、曲線、矩陣與流程為主，因此修正採 Canvas 2D；沒有需要深度或三維旋轉的場景。

## 共用改善

- 手機保留最小 640px 畫布，提供場景內左右捲動、鍵盤焦點與操作提示；資料切分策略採獨立直向版。
- 放大畫面維持正確比例；手機的橫向滑動與滑桿定位可並用。
- 播放器支援依設定動態計算單步數，讓批次、模型與折數的步進一致。
- 保留原教材對照圖；HTML 流程摘要正確處理 Mermaid 的 `<br>`／`<br/>`。

## 驗證

- 瀏覽器逐場景擷取起點、中段與終點，逐一目視檢查；另檢查全部 42 個手機畫布。
- 操作 189 組控制項設定，涵蓋下拉選項、數值極值與開關；全部場景檢查播放／暫停、單步、重設、放大與 Esc 返回。
- 修正後檢查沒有執行錯誤或文字越出畫布；最後調整的 7 個場景另行複查。
- 全部 42 個手機場景確認可在場景內捲動、頁面寬度不溢出；放大畫布比例正確，Esc 可返回。
- `node docs/glossary/check-scenes.mjs`：42 個場景、1,561 個狀態，涵蓋數值有效性、資料不重複分配、時間順序、每折進度與批次計數。
- `node docs/glossary/check-math.mjs`：切分、回歸、梯度、分類指標、ROC、R²、KNN、決策樹與聚類檢查通過。
- `.venv/bin/python docs/glossary/build.py`：重新產生 12 章靜態頁面與 42 個動畫入口。

## 逐場景結果

| 章節 | 場景 | 處理與結果 |
| :--- | :--- | :--- |
| 01 機器學習的類型 | [監督式：分類與回歸](./01-learning-paradigms/index.html#supervised) | 檢查分類／回歸與新房屋面積極值；共用手機畫布改善。 檢查通過。 |
| 01 機器學習的類型 | [非監督式：聚類與降維](./01-learning-paradigms/index.html#unsupervised) | 增加群組形狀與圖例；投影限定在座標區內，避免極端角度越界。 檢查通過。 |
| 01 機器學習的類型 | [學習類型比較](./01-learning-paradigms/index.html#paradigm-comparison) | 分開強化學習回饋說明與樣本方塊，避免重疊。 檢查通過。 |
| 02 數據結構 | [特徵與資料型態](./02-data-structures/index.html#features) | 五種型態逐一檢查；共用手機畫布改善。 檢查通過。 |
| 02 數據結構 | [標籤與任務](./02-data-structures/index.html#labels) | 重排輸入卡與答案欄，讓移動標籤不遮住特徵。 檢查通過。 |
| 02 數據結構 | [特徵矩陣 X 與標籤 y](./02-data-structures/index.html#matrix) | 欄名隨矩陣一起移動；選定交叉格增加獨立外框。 檢查通過。 |
| 03 數據分割 | [訓練／測試分割](./03-data-splitting/index.html#train-test) | 加入 ID、圖例與來源位置；改為逐筆移動。 檢查通過。 |
| 03 數據分割 | [三份資料的角色](./03-data-splitting/index.html#train-validation-test) | 加入 ID、來源對照、逐筆分配與用途箭頭。 檢查通過。 |
| 03 數據分割 | [隨機、分層與時間切分](./03-data-splitting/index.html#split-strategies) | 保留完整原始資料；抽樣與分配分階段、逐筆移動，標示比例與時間分界；手機採直向版。 檢查通過。 |
| 03 數據分割 | [資料洩漏與診斷](./03-data-splitting/index.html#leakage) | 把所選診斷情境顯示於畫布，與計算統計量分開。 檢查通過。 |
| 04 特徵工程 | [特徵選擇](./04-feature-engineering/index.html#selection) | 包裝法改為逐組線性模型訓練與驗證 MSE；嵌入法實算 Lasso 正則化路徑、係數長條與非零保留欄位。檢查全部保留數、L1 強度端點、單步計數、播放／重設／放大及手機捲動。 檢查通過。 |
| 04 特徵工程 | [特徵縮放](./04-feature-engineering/index.html#scaling) | Canvas 改為面積與房間數的兩候選近鄰比較，分色呈現平方距離貢獻，四步展示縮放與排名；檢查三種縮放、單位換算不變性、離群值及手機操作。 檢查通過。 |
| 04 特徵工程 | [類別編碼](./04-feature-engineering/index.html#encoding) | One-Hot 起點方塊分開，補上對應類別欄名。 檢查通過。 |
| 05 模型參數 | [模型參數](./05-model-parameters/index.html#parameters) | 檢查 w、b 極值與梯度更新；共用手機畫布改善。 檢查通過。 |
| 05 模型參數 | [超參數](./05-model-parameters/index.html#hyperparameters) | 縮小樹的分支間距，避免最大深度 4 越界或覆蓋損失圖。 檢查通過。 |
| 05 模型參數 | [比較與搜尋](./05-model-parameters/index.html#parameter-search) | 檢查三種搜尋順序；共用手機畫布改善。 檢查通過。 |
| 06 優化算法 | [梯度下降](./06-optimization/index.html#gradient-descent) | 切線與負梯度箭頭限制在座標區內。 檢查通過。 |
| 06 優化算法 | [學習率](./06-optimization/index.html#learning-rate) | 檢查學習率極值與發散邊界呈現；共用手機畫布改善。 檢查通過。 |
| 06 優化算法 | [Batch／SGD／Mini-batch](./06-optimization/index.html#batch-size) | 檢查 SGD／Mini-batch／Batch 與批次極值；共用手機畫布改善。 檢查通過。 |
| 06 優化算法 | [優化器補充](./06-optimization/index.html#optimizers) | 檢查三種更新公式、學習率極值；共用手機畫布改善。 檢查通過。 |
| 07 常見算法 | [KNN](./07-common-algorithms/index.html#knn) | 圓／方／三角形區分類別；查詢點用空心菱形；距離改為排名表，每一步增加一位鄰居。 檢查通過。 |
| 07 常見算法 | [決策樹](./07-common-algorithms/index.html#decision-tree) | 從未切分開始逐層揭露；單步數依深度變動，補上 A／B 圖例。 檢查通過。 |
| 07 常見算法 | [SVM](./07-common-algorithms/index.html#svm) | 檢查線性與徑向模式；徑向邊界改成隨時間描繪。 檢查通過。 |
| 07 常見算法 | [樸素貝氏](./07-common-algorithms/index.html#naive-bayes) | 同時呈現兩類條件機率、計算路徑、原始分數與正規化比例；單步納入一個詞。 檢查通過。 |
| 08 集成學習 | [Bagging](./08-ensemble-learning/index.html#bagging) | 修正預測標記的折返路徑；展示 Bootstrap ID 與重複樣本；每一步整合一個模型。 檢查通過。 |
| 08 集成學習 | [Boosting](./08-ensemble-learning/index.html#boosting) | 檢查輪數與步長極值、殘差與樹樁數值；共用手機畫布改善。 檢查通過。 |
| 08 集成學習 | [Stacking](./08-ensemble-learning/index.html#stacking) | 每一步完成一折；全部 OOF 完成後開啟元模型；標示表格僅預覽 9／24 筆。 檢查通過。 |
| 09 評估指標 | [混淆矩陣](./09-evaluation-metrics/index.html#confusion-matrix) | 標示真實／預測兩軸、樣本 ID 與類別圖例；逐筆移動，保留來源位置。 檢查通過。 |
| 09 評估指標 | [分類指標與 ROC](./09-evaluation-metrics/index.html#classification-metrics) | 檢查門檻 0／1、正類比例極值與未定義指標；共用手機畫布改善。 檢查通過。 |
| 09 評估指標 | [MAE、MSE、RMSE](./09-evaluation-metrics/index.html#regression-errors) | 重排平方色塊與說明，增加 |e|、e²、倍率與圖例，修正文字裁切。 檢查通過。 |
| 09 評估指標 | [R²](./09-evaluation-metrics/index.html#r-squared) | 補上模型／平均基準／殘差圖例與平方和比例尺說明。 檢查通過。 |
| 10 模型性能問題 | [欠擬合、良好擬合與過度擬合](./10-model-performance/index.html#fit-spectrum) | 驗證資料更正為紫色菱形，加入訓練／驗證／模型／真實規律圖例。 檢查通過。 |
| 10 模型性能問題 | [偏差與變異數](./10-model-performance/index.html#bias-variance) | 檢查重抽樣、模型次數與平均模型開關；共用手機畫布改善。 檢查通過。 |
| 10 模型性能問題 | [學習曲線與早停](./10-model-performance/index.html#learning-curves) | 檢查早停／繼續與最佳點；共用手機畫布改善。 檢查通過。 |
| 10 模型性能問題 | [改善方式](./10-model-performance/index.html#solutions) | 驗證資料更正為紫色菱形，加入模型與資料圖例。 檢查通過。 |
| 11 模型泛化 | [未見資料與泛化](./11-generalization/index.html#unseen-data) | 驗證資料更正為紫色菱形，加入模型與資料圖例。 檢查通過。 |
| 11 模型泛化 | [泛化鴻溝](./11-generalization/index.html#generalization-gap) | 統一 0–1 誤差尺度；差距數字與動畫中實際展示值同步。 檢查通過。 |
| 11 模型泛化 | [資料分割與 K 折交叉驗證](./11-generalization/index.html#cross-validation) | 每一步完成一折；起點不預先揭露分數，平均與標準差按已完成折數計算。 檢查通過。 |
| 11 模型泛化 | [提升泛化](./11-generalization/index.html#improve-generalization) | 驗證資料更正為紫色菱形，加入模型與資料圖例。 檢查通過。 |
| 12 機器學習訓練過程 | [端到端生命週期](./12-training-process/index.html#lifecycle) | 改為蛇形流程；箭頭連接卡片邊緣，監控形成真正的重訓閉環，每格列出產出。 檢查通過。 |
| 12 機器學習訓練過程 | [訓練迭代與 Epoch／Batch／Iteration](./12-training-process/index.html#inner-loop) | 每一步完成一次實際 Iteration；計數、批次 ID、Epoch 與結束狀態一致，處理無完整批次。 檢查通過。 |
| 12 機器學習訓練過程 | [健康度與檢查清單](./12-training-process/index.html#training-health) | 誤差使用共用 0–1 尺度；三個流程檢查燈補上各自名稱。 檢查通過。 |

畫面與機器檢查資料放在專案的 `output/playwright/glossary-audit/`，供本機複核。
