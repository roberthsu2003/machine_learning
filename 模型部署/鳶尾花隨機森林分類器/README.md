# 🌸 機器學習模型部署：FastAPI + Gradio 雙介面全生命週期教學講義

本專案是一個特別為機器學習教學設計的**實務部署範例**。它展示了從**模型訓練、序列化儲存、建立 RESTful API，到設計網頁互動 UI**，最後一鍵部署至免費雲端平台（如 Render / Hugging Face Spaces）的完整生命週期。

---

## 🎯 本課教學目標

學習完本專案，你將掌握以下核心技術：
1. **模型序列化**：學會使用 `joblib` 將 scikit-learn 隨機森林模型與其類別名稱打包序列化，並了解反序列化的安全風險。
2. **高效 API 開發**：使用 `FastAPI` 建立具備 `Pydantic` 強型別校驗與資料過濾機制的預測端點。
3. **線上熱重載**：設計 `POST /train` 端點，在線上動態訓練模型並即時更新伺服器記憶體中的模型實例。
4. **低代碼 UI 設計**：利用 `Gradio` 的 Blocks 佈局，打造擁有指標卡與特徵重要性橫條圖的高質感網頁介面。
5. **雲端網路優化**：理解反向代理（Reverse Proxy）下的緩衝與連線佇列問題，學會透過 `queue=False` 與 `.release()` 優化服務。

---

## 📂 專案架構與檔案說明

本專案目錄下包含以下核心檔案：
```text
鳶尾花隨機森林分類器/
├── app.py                             # FastAPI + Gradio 融合服務（啟動主程式）
├── train_save.py                      # 模型訓練與元數據序列化腳本
├── practice_train_api_answer.ipynb    # 🎓 拆解範例 1：隨機森林訓練與 API 端點邏輯
├── practice_predict_api_answer.ipynb  # 🎓 拆解範例 2：花朵特徵預測與 API 端點
├── practice_gradio_ui_answer.ipynb    # 🎓 拆解範例 3：Gradio 網頁 UI 與前後端整合
├── requirements.txt                   # 相依套件清單
├── deploy.sh                          # Hugging Face 一鍵部署指令檔
├── render.yaml                        # Render 雲端部署配置檔
└── README.md                          # 本教學講義文件
```

> [!TIP]
> **💡 給學生的學習提醒：從步驟拆解 Jupyter Notebook 入手**
> 
> 主程式 `app.py` 整合了模型訓練、FastAPI RESTful 端點以及 Gradio Web 介面，程式碼架構完整且檔案較大，初學者可能較難一口氣理解全貌。
> 
> 為降低學習門檻，本專案特別提供了 **3 個步驟式拆解範例 (`.ipynb`)**，建議學生可先開啟並依序學習：
> 1. **`practice_train_api_answer.ipynb`**：拆解學習如何訓練隨機森林分類器、序列化保存模型與實作 `/train` API。
> 2. **`practice_predict_api_answer.ipynb`**：拆解學習花朵特徵輸入驗證、機率計算（`predict_proba`）與實作 `/predict` API。
> 3. **`practice_gradio_ui_answer.ipynb`**：拆解學習如何單獨建構 Gradio 互動式 UI 介面（含品種預測卡片與機率分析）並與後端 API 串接。
> 
> 先透過 Notebook 掌握各模組細節後，再閱讀完整融合的主程式 `app.py`，會更容易融會貫通！

---

## 📖 核心觀念解析

### 1. 什麼是模型部署 (Model Deployment)？
在開發環境訓練好的模型只存在於記憶體中，關閉程式後便消失。**部署**就是將模型移至雲端伺服器上長期運行。我們不可能在使用者每次要預測時都重跑一次訓練（那會耗費數秒到數分鐘），而是需要**將訓練好的模型存檔，並在伺服器啟動時直接讀入**。

### 2. 什麼是序列化 (Serialization)？
*   **序列化 (Serialization)**：將記憶體中的模型物件，儲存為硬碟中的二進位檔案（例如 `iris_model.joblib`）。
*   **反序列化 (Deserialization)**：伺服器讀取此二進位檔案，在毫秒內還原成記憶體中的模型物件，用以進行「即時預測 (Inference)」。
*   > [!CAUTION]
    > **安全性警示**：`joblib` 或 `pickle` 在反序列化時會執行任意程式碼。**千萬不要**在伺服器上載入來源不明的模型檔案，否則將使伺服器面臨被控或入侵的安全風險！

### 3. Gradio 與 FastAPI 的融合掛載
Gradio 框架的底層本來就是以 **FastAPI** 寫成。透過 Gradio 提供的 `gr.mount_gradio_app(app, demo, path="/")` 函數，我們能將 Gradio 網頁介面作為路由掛載到我們自己定義的 FastAPI 實例上。
這樣一來，你的服務既是網頁（訪問 `/`），也是高效能的 API（訪問 `/predict`、`/train`），且無需為這兩個服務維護多個連接埠！

---

## 🛠️ 本地實作指引

### 1. 環境準備與套件安裝
建議在虛擬環境（如 `.venv`）下執行，以避免套件衝突：
```bash
# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 啟動服務
主程式 `app.py` 設有**自動化雙保險設計**：啟動時若在目錄下找不到模型檔 `iris_model.joblib`，會自動呼叫 `train_save.py` 進行訓練並存檔，保證服務絕不因缺少模型檔而崩潰。
```bash
# 本地直接啟動
python app.py
```
* **熱重載模式**：開發時如需修改代碼自動重啟，請使用：
  `RELOAD=true python app.py`
* **自訂 Port 啟動**：
  `PORT=3000 python app.py`

### 3. 本地測試與驗證

#### 🅰️ 測試網頁 UI
打開瀏覽器，訪問 `http://127.0.0.1:8000/`：
* 在 **「🔮 即時模型預測」** 分頁中調整特徵滑桿，並觀察預測卡片的背景色與機率長條圖。
* 前往 **「⚙️ 線上模型訓練與評估」** 分頁，調整樹數量等參數後點擊「開始訓練」，查看指標卡片的變化。

#### 🅱️ 測試 FastAPI RESTful API 端點
開啟終端機，執行以下 `curl` 指令來測試我們的自訂 API 端點：

* **即時預測端點：`POST /predict`**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
  ```
  *回傳格式*：
  ```json
  {"prediction_id":0,"prediction_label":"setosa","probabilities":{"setosa":1.0,"versicolor":0.0,"virginica":0.0}}
  ```

* **線上訓練端點：`POST /train`**
  ```bash
  curl -X POST http://127.0.0.1:8000/train \
    -H "Content-Type: application/json" \
    -d '{"n_estimators": 50, "max_depth": 5, "test_size": 0.3, "random_state": 42}'
  ```
  *回傳格式*：
  ```json
  {"status":"success","accuracy":1.0,"train_time":0.0164,"feature_importances":{"sepal length":0.11,"sepal width":0.04,"petal length":0.42,"petal width":0.43},"message":"模型訓練完成並儲存成功！"}
  ```

* **API 自動文件：`/docs`**
  打開瀏覽器造訪 `http://127.0.0.1:8000/docs`，即可在 Swagger UI 進行互動式 API 測試。

---

## 🎯 雲端部署指引

### 1. 部署到 Render
1. 將專案推送到您的 GitHub 倉庫。
2. 登入 Render，建立一個新的 **Web Service**，並連結該 GitHub 倉庫。
3. 填寫以下設定：
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `python app.py`
4. 服務啟動後，Render 會自動注入環境變數 `PORT`，程式會自動讀取並綁定。

### 2. 部署到 Hugging Face Spaces (Gradio SDK)
由於 Hugging Face Spaces 會讀取最外層目錄的 `app.py`。如果您是在多專案的子資料夾下，可以使用專案內附的 `deploy.sh` 進行一鍵打包上傳：
```bash
# 執行部署指令檔，並依提示輸入 HF 帳號、Space 名稱與 Access Token (Write 權限)
./deploy.sh
```

---

## 💡 進階專題：針對雲端反向代理與效能優化說明

在免費的雲端主機（如 Render Free，僅 0.1 CPU/512MB RAM）或有反向代理（Reverse Proxy）的網頁環境中，使用 Gradio 常會遇到**畫面變灰、死當**或不斷卡在 `queue: 1/1` 的情況。本專案透過以下設計解決了這些痛點：

1. **滑桿事件改用 `.release()` 代替 `.change()`**
   * *問題*：`.change()` 會在拖動滑桿的過程中，每一像素的改變都發送一次請求，短短一秒湧入數十個 API 呼叫，瞬間卡死免費主機的 CPU。
   * *解決方案*：使用 `.release()`。只有當使用者**放開**滑桿時才發送一次預測請求，請求量減少 95% 以上，完全避免 CPU 飆滿排隊。
2. **禁用 SSE 佇列 (`queue=False`)**
   * *問題*：Gradio 預設使用 SSE (Server-Sent Events) 的長連線佇列。Render 這類反向代理會緩衝長連線，導致資料被攔截，前端畫面卡在 `queue: 1/1` 一直計時，即使後端早已運算完畢。
   * *解決方案*：所有 Gradio 事件設定 `queue=False`，使其改走標準短連線 HTTP POST，即時回傳。
3. **隱藏載入遮罩動畫 (`show_progress="hidden"`)**
   * *問題*：輸出元件更新時，預設會整欄蓋上灰色遮罩，在有網路延遲的環境下會造成嚴重的視覺閃爍卡頓。
   * *解決方案*：在事件中設定 `show_progress="hidden"`，使機率長條圖與預測卡片的更新非常滑順，完全沒有閃爍與卡頓感。

---

## 📝 課後習題與延伸思考

1. **特徵範圍驗證**：在 `app.py` 中，Pydantic 模型限制特徵值必須在 `0.1` 與 `10.0` 之間。若透過 `curl` 傳送非法特徵（例如 `15.0`），FastAPI 會返回什麼 HTTP 狀態碼與錯誤訊息？
2. **特徵重要性分析**：請觀察訓練分頁產生的特徵重要性橫條圖，哪一項特徵對隨機森林模型分類鳶尾花影響最大？這與特徵本身的生物學意義有何關係？
3. **動態更新驗證**：如果我們在訓練分頁將隨機森林的樹數量（`n_estimators`）改為 `10`，接著切換回預測分頁進行預測，模型是否能直接使用這棵「只有 10 棵樹的隨機森林」？背後的全域變數是如何動態重載的？

---

## 授權

MIT License
