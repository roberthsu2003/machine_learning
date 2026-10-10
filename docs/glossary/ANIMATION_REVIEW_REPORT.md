# 動畫檢查與改善報告

- 檢查依據：`docs/glossary/ANIMATION_REVIEW_PROMPT.md`
- 檢查日期：2026-10-10
- 實際清點：12 章 42 個場景（`chapters.json` 與頁面 `data-scene` 一致）＋ 簡報頁 `docs/index.html` 6 個 motion 動畫，共 48 個。
- 狀態圖例：✅ 已實際測過 ｜ 🔧 已修正並複測 ｜ ⚠️ 尚未驗證（本報告不把未檢查場景標為通過）

## 驗證環境與指令

- 瀏覽器：Playwright Chromium Headless Shell 156.0.8078.4（桌面 1440×900、手機 390×844）
- 本機服務：`python3 -m http.server 8765 --directory docs`
- 產生器：`.venv/bin/python docs/glossary/build.py` → `Built 12 chapter pages, index, and 42 animation mounts.` ✅
- 數值檢查：`node docs/glossary/check-math.mjs` → 修正前後皆通過 ✅
- 自動化操作（每場景）：讀取初始訊息 → 按「下一步」→ 播放約 0.6 秒後暫停 → 時間軸拖到終點 → 讀 `data-result` metrics → 按「重設」。全程監聽 console error 與 pageerror，結果為 0 錯誤 ✅
- 截圖：`output/animation-review/`（12 張章節 1440px ＋ 01、09 章 390px ＋ 簡報首頁）
- 獨立核算：以 `shared/math.js` 以外的手算路徑複算 Ch01 回歸線（w≈49.64、b≈17.86）、Ch09 預設斜率 MAE=0.34／MSE=0.178／RMSE=0.422，與頁面 `data-result` 一致 ✅

## 本次修正總覽（🔧）

1. **分類散佈圖 Y 軸誤標為 `y`**：`01-unsupervised`、`07-knn`、`07-decision-tree`、`07-svm` 的座標兩軸都是特徵，Y 軸卻顯示預設標籤 `y`，違反「分類圖兩特徵軸不得把其中一軸標成標籤 y」。已改為 `特徵 X₁`／`特徵 X₂`。
2. **梯度下降座標軸無意義**：`06` 章 `ball()` 共用函式沿用預設 `x`／`y` 標籤，橫軸實為參數 θ、縱軸為損失 J。已改為 `參數 θ`／`損失 J(θ)`（含三球對照小圖）。
3. **特徵縮放右軸顯示英文代碼**：`04-scaling` 右軸 `ylabel` 直接使用 `raw`／`z`／`minmax` 代碼。已改為 `原始值`／`Z-score 值`／`Min-Max 值`，左軸補上單位 `原始面積（坪）`。
4. **誤差平方塊暗中封頂**：`09-regression-errors` 以 `min(75, e*16)` 截斷邊長，卻聲稱「邊長對應 |誤差|」。已改為無截斷等比縮放 `邊長 = 8 × |誤差|`（最大誤差 7.2 → 57.6px，符合 72px 格子），圖說與訊息同步明示比例；同時補上 `特徵 X₁`／`標籤 y` 軸標。
5. **泛化鴻溝浮點殘留**：`11-generalization-gap` 顯示「差 2.9999999999999973%」。已改為整數百分點表述。
6. **回歸場景軸標補強**：`05-parameters`、`08-boosting`、`09-r-squared`、`10-bias-variance` 及 `shared/fitting.js` 的回歸座標補上 `特徵 X₁`／`標籤 y`（原為預設 `x`／`y` 小寫且無解釋）。

## 逐場景檢查表

| 章節／頁面 | 場景 ID 與網址 | 教學目標 | 發現的問題 | 修正內容 | 驗證方法與結果 | 尚未解決事項 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 機器學習的類型 | supervised `#supervised` ／ `glossary/01-learning-paradigms/#supervised` | 區分已知標籤 y 與預測 ŷ；三階段看資料→建模型→預測 | ✅ 已知問題（雙軸、y 軸混淆、流程不清）前次已修正；本次複測流程正確 | 無（僅複測） | ✅ 自動化三階段訊息切換正確；獨立複算回歸線 w≈49.64 一致 | 無 |
| 01 機器學習的類型 | unsupervised `#unsupervised` | K-Means 指派→更新中心；投影 2D→1D 捨棄資訊 | 🔧 散佈軸標為 x／y，Y 軸易誤認為標籤 | 軸改 `特徵 X₁`／`特徵 X₂` | 🔧 複測：播放／單步／重抽正常，0 錯誤 | ⚠️ 投影角滑桿極值視覺尚未逐一截圖比對 |
| 01 機器學習的類型 | paradigm-comparison `#paradigm-comparison` | 對照有／無標籤管線；半監督與 RL 僅示意 | ✅ 示意標示清楚（訊息註明「僅示意」） | 無 | ✅ 四種模式切換＋單步正常 | 無 |
| 02 數據結構 | features `#features` | 特徵拆解；連續／離散／有序／無序／非結構化 | ⚠️ `資料型態` 參數只改變文字說明，不改變幾何 | 無（記錄為限制） | ✅ 操作無錯誤；文字隨參數同步 | 型態切換若能改變圖形形狀會更直觀，留待後續 |
| 02 數據結構 | labels `#labels` | X 是輸入、y 是答案；分類／回歸標籤型態 | ✅ 分類門檻（>1500 萬=高價）已明示為演示定義 | 無 | ✅ 切換＋顯示答案同步 | 無 |
| 02 數據結構 | matrix `#matrix` | X 形狀 (3,4)、y 長度 3；列是樣本、欄是特徵 | ✅ 訊息直接給出 `X[i,j]` 取值 | 無 | ✅ 選列／選欄＋重播正常 | 無 |
| 03 數據分割 | train-test `#train-test` | 同一樣本只能進一份資料；互斥分割 | ✅ 交集 0 筆訊息明確；check-math 覆蓋互斥 | 無 | ✅ 比例滑桿＋重抽正常 | 無 |
| 03 數據分割 | train-validation-test `#train-validation-test` | Train 學參數／Validation 調參／Test 封存 | ✅ Test 不參與訊息明確 | 無 | ✅ 三角色切換正常 | 無 |
| 03 數據分割 | split-strategies `#split-strategies` | 隨機／分層／時間切分的類別比例差異 | ✅ 分層受整數限制已註明 | 無 | ✅ 三模式＋比例正常 | 無 |
| 03 數據分割 | leakage `#leakage` | 縮放器只 fit Train；洩漏路徑對照 | ✅ 橘色洩漏路徑與正確流程對照清楚 | 無 | ✅ 對照切換正常 | 無 |
| 04 特徵工程 | selection `#selection` | 過濾實算 \|r\| 選特徵；包裝／嵌入僅示意 | ✅ 示意邊界已註明（未執行 RFE／Lasso） | 無 | ✅ 保留數＋方法切換正常 | 無 |
| 04 特徵工程 | scaling `#scaling` | 縮放前後位置變化；離群值影響 μ／σ | 🔧 右軸 ylabel 為英文代碼；左軸缺單位 | 中文軸名＋單位（見總覽 3） | 🔧 複測：三模式＋離群值切換正常，數值與訊息一致 | ⚠️ 手機版雙座標在 390px 較擠，僅截圖確認可讀，建議後續加直式排列 |
| 04 特徵工程 | encoding `#encoding` | One-Hot／有序整數／無序整數陷阱 | ✅ 陷阱模式已警告「假的大小關係」 | 無 | ✅ 三模式＋類別切換正常 | 無 |
| 05 模型參數 | parameters `#parameters` | w 斜率、b 截距；梯度更新實算 | 🔧 座標為預設 x／y 小寫且無解釋 | 補 `特徵 X₁`／`標籤 y` | 🔧 複測：w／b 滑桿＋20 步更新正常 | 無 |
| 05 模型參數 | hyperparameters `#hyperparameters` | 超參數（深度、α）vs 學出參數 | ✅ 右圖 J(w) 軸標正確；訊息註明非學出設定 | 無 | ✅ 深度＋α 操作正常 | 無 |
| 05 模型參數 | parameter-search `#parameter-search` | 網格／隨機／貝氏示意；驗證分數選最佳 | ✅ 貝氏僅排序示意已註明 | 無 | ✅ 12 組逐步揭露＋最佳標示正常 | 無 |
| 06 優化算法 | gradient-descent `#gradient-descent` | θ ← θ − α·2θ；切線與負梯度箭頭 | 🔧 座標軸為 x／y（應為 θ／J） | 改 `參數 θ`／`損失 J(θ)` | 🔧 複測：起點拖曳＋α＋20 步正常 | 無 |
| 06 優化算法 | learning-rate `#learning-rate` | 小／中／大步長對照；α=1.1 發散 | 🔧 小圖軸標缺失（沿用 x／y） | 三小圖補 θ／J(θ) 標 | 🔧 複測：α 滑桿＋16 步正常 | 無 |
| 06 優化算法 | batch-size `#batch-size` | 批次→平均梯度→一次更新；計數規則 | ✅ ceil／drop_last 說明已具備 | 無 | ✅ 三模式＋Batch Size＋重抽正常 | 無 |
| 06 優化算法 | optimizers `#optimizers` | SGD／Momentum／Adam 同一起點對照 | ✅ 已註明單一地形不能證明優劣 | 無 | ✅ 切換＋α 正常 | 無 |
| 07 常見算法 | knn `#knn` | 最近 k 鄰居投票；距離同步 | 🔧 座標軸 x／y 混淆；另搜尋圓半徑未裁剪 | 軸改 X₁／X₂＋類別圖例提示 | 🔧 複測：拖曳＋k＋距離顯示正常 | 大距離查詢點的搜尋圓可能超出座標，僅描邊溢出，數值正確，留待後續裁剪 |
| 07 常見算法 | decision-tree `#decision-tree` | 切割線↔樹路徑對應；Gini 實算 | 🔧 座標軸 x／y 混淆 | 軸改 X₁／X₂ | 🔧 複測：深度＋查詢點正常 | 無 |
| 07 常見算法 | svm `#svm` | 間隔與支援向量；核技巧僅示意 | 🔧 座標軸 x／y 混淆 | 軸改 X₁／X₂ | 🔧 複測：線性／核＋邊界滑桿正常 | 無 |
| 07 常見算法 | naive-bayes `#naive-bayes` | 先驗×條件機率→相對分數 | ✅ Bernoulli 簡化假設已註明 | 無 | ✅ 詞彙勾選＋先驗正常 | 無 |
| 08 集成學習 | bagging `#bagging` | Bootstrap 有放回→獨立模型→投票／平均 | ✅ 唯一樣本數揭露重複抽樣；同票規則已註明 | 無 | ✅ 模型數＋整合方式＋重抽正常 | 無 |
| 08 集成學習 | boosting `#boosting` | 逐輪學殘差；η 加權 | 🔧 座標為預設 x／y | 補 X₁／y | 🔧 複測：輪數＋η 正常 | 無 |
| 08 集成學習 | stacking `#stacking` | OOF 重排→元模型；不洩漏 | ✅ 「?」未揭露機制與 Ridge 權重顯示清楚 | 無 | ✅ K 折＋逐步揭露正常 | 無 |
| 09 評估指標 | confusion-matrix `#confusion-matrix` | TP／FP／FN／TN 落格；門檻定義正類 | ✅ 方塊保留真實標籤色彩 | 無 | ✅ 門檻滑桿＋落格正常；TP=9／FP=3／FN=0／TN=18（預設） | 無 |
| 09 評估指標 | classification-metrics `#classification-metrics` | 門檻→四格＋指標＋ROC 同步；AUC 可<0.5 | ✅ 分母為零顯示未定義 | 無 | ✅ 門檻＋正類比例正常；AUC=0.899（預設） | 無 |
| 09 評估指標 | regression-errors `#regression-errors` | MAE／MSE／RMSE；平方懲罰面積 | 🔧 平方塊暗中封頂（見總覽 4） | 等比縮放＋明示比例＋軸標 | 🔧 複測：斜率＋離群值正常；獨立複算 MAE=0.34／MSE=0.178 一致；最大邊 57.6px 無裁切 | 無 |
| 09 評估指標 | r-squared `#r-squared` | R²=1−SSE／SST；負 R²；常數 y 未定義 | 🔧 座標為預設 x／y | 補 X₁／y | 🔧 複測：四模式正常 | 無 |
| 10 模型性能問題 | fit-spectrum `#fit-spectrum` | 容量→欠／良／過擬合；訓練／驗證 MSE | ✅ 曲線插值已註明非訓練迭代 | 無 | ✅ 次數＋驗證揭露正常 | 無 |
| 10 模型性能問題 | bias-variance `#bias-variance` | 8 組重抽；偏差平方 vs 變異數 | 🔧 座標為預設 x／y | 補 X₁／y | 🔧 複測：次數＋平均模型＋重抽正常 | 無 |
| 10 模型性能問題 | learning-curves `#learning-curves` | 早停；驗證最佳點 | ✅ 預設示意資料已註明 | 無 | ✅ 早停勾選＋30 步正常；最佳 Epoch 9 | 無 |
| 10 模型性能問題 | solutions `#solutions` | 資料／容量／正則化對照 | ✅ 不保證每次改善已註明 | 無 | ✅ 四策略正常 | 無 |
| 11 模型泛化 | unseen-data `#unseen-data` | 新樣本揭露；高容量敏感度 | ✅ 高容量≠記憶已註明 | 無 | ✅ 兩模式＋逐步揭露正常 | 無 |
| 11 模型泛化 | generalization-gap `#generalization-gap` | 鴻溝小≠好；看絕對高度 | 🔧 浮點殘留 2.9999% | 整數百分點 | 🔧 複測：三情境正常 | 無 |
| 11 模型泛化 | cross-validation `#cross-validation` | 先封存 Test 再 K 折輪替；均值±標準差 | ✅ Test 封存 6 筆明確 | 無 | ✅ K＋次數＋逐步揭露正常 | 無 |
| 11 模型泛化 | improve-generalization `#improve-generalization` | 資料／容量／正則／早停示意 | ✅ 早停為保守擬合代表已註明 | 無 | ✅ 五策略正常 | 無 |
| 12 訓練過程 | lifecycle `#lifecycle` | 六階段＋監控回饋閉環 | ✅ 切分／fit 順序已註明 | 無 | ✅ 階段＋回饋勾選正常 | 無 |
| 12 訓練過程 | inner-loop `#inner-loop` | Batch／Iteration／Epoch；ceil 與 drop_last | ✅ ceil(N/B) 與丟棄筆數同步顯示 | 無 | ✅ N／Batch／Epoch＋drop_last 正常 | 無 |
| 12 訓練過程 | training-health `#training-health` | 預設情境＋三關卡檢查 | ✅ 預設情境已註明 | 無 | ✅ 狀態＋三勾選正常 | 無 |
| 簡報教材 | flow（`docs/index.html#chapter-5`） | 資料→模型→預測流程四階段 | ✅ 每階段圖說完整 | 無 | ✅ 播放／重播／單步／時間軸正常 | ⚠️ 與 glossary 播放器行為小差異（此處重播自動播放）屬既有設計，未改 |
| 簡報教材 | network（`#chapter-6`） | 前向傳播訊號逐層傳遞 | ✅ 圖例（光點／神經元／權重）完整 | 無 | ✅ 同上 | 無 |
| 簡報教材 | features（`#chapter-7`） | 線條→零件→整車層級示意 | ✅ 概念示意已註明 | 無 | ✅ 同上 | 無 |
| 簡報教材 | training（`#chapter-11`） | 前向→損失→反傳→更新；logistic 實算 | ✅ 單參數簡化已註明 | 無 | ✅ 訓練 1 次／10 次／重設＋損失連動正常（代碼審查） | ⚠️ 未做逐鍵自動化點擊，僅代碼與畫面審查 |
| 簡報教材 | regression（`#chapter-12`） | 斜率／截距→MSE；最小平方解 | ✅ 參數插值非梯度求解已註明 | 無 | ✅ 斜率／截距／坪數滑桿＋MSE 連動（代碼審查） | 同上 |
| 簡報教材 | inference（`#chapter-14`） | 教學公式推論；非訓練所得 | ✅ 假設公式與 735 萬重現已註明 | 無 | ✅ 表單輸入＋執行推論＋重設（代碼審查） | 同上 |

## 待提交檔案（未自行 commit／push）

- `docs/glossary/01-learning-paradigms/scenes/index.js`
- `docs/glossary/04-feature-engineering/scenes/index.js`
- `docs/glossary/05-model-parameters/scenes/index.js`
- `docs/glossary/06-optimization/scenes/index.js`
- `docs/glossary/07-common-algorithms/scenes/index.js`
- `docs/glossary/08-ensemble-learning/scenes/index.js`
- `docs/glossary/09-evaluation-metrics/scenes/index.js`
- `docs/glossary/10-model-performance/scenes/index.js`
- `docs/glossary/11-generalization/scenes/index.js`
- `docs/glossary/shared/fitting.js`
- `docs/glossary/ANIMATION_REVIEW_REPORT.md`（本報告）
- 截圖：`output/animation-review/`（15 張 PNG）

## 限制與後續建議

1. 觸控拖曳（KNN／梯度起點）僅做代碼審查（pointer 事件具備 `setPointerCapture`），未在實體觸控裝置驗證。
2. `02-features` 的資料型態切換僅改文字，建議後續以形狀變化對應五種型態。
3. KNN 搜尋圓在大距離時可能描邊溢出座標，數值正確，建議後續加 `clip()`。
4. 簡報頁三個表單型互動（訓練／回歸／推論）因屬既有 `course.js`/`motion.js` 體系，僅做代碼與畫面審查，未納入本次自動化點擊矩陣。
5. 章序、模組、網址與場景錨點均未變動；Light Mode、繁體中文維持；未新增任何訓練成效宣稱。
