# 機器學習模型部署：免 Dockerfile 的 FastAPI + Gradio 混合服務部署教學

在機器學習專案中，當我們訓練好模型後，通常會面臨兩個開發需求：
1.  **建立 Web API (例如 FastAPI)**：提供一個高效、規格化的預測端點，讓外部程式（如 App 或其他伺服器）能透過發送 JSON 請求來獲取預測結果。
2.  **建立 網頁 UI 介面 (例如 Gradio)**：提供一個簡單、直觀的網頁，讓非程式背景的客戶或同學可以直接在瀏覽器上操作（如調整滑桿、按按鈕）並即時看到結果。

以往要部署這兩套系統，可能需要撰寫複雜的 **Dockerfile** 將其容器化。**本教學將介紹一個極為巧妙的「免 Dockerfile」方案**：利用 Gradio 底層本來就是 FastAPI 的特性，在 Gradio 的 Space 中直接掛載 FastAPI 端點。

這樣一來，學生**既能完整學習到 FastAPI 與 Pydantic 的 API 開發，又不需要學習 Docker 觀念**，還能同時擁有一套免費且漂亮的網頁介面！

---

## 目錄
1. [基本觀念：模型序列化](#一-基本觀念模型序列化)
2. [技術架構：Gradio 結合 FastAPI 的巧妙之處](#二-技術架構gradio-結合-fastapi-的巧妙之處)
3. [專案架構與程式碼說明](#三-專案架構與程式碼說明)
4. [本地測試步驟](#四-本地測試步驟)
5. [部署至 Hugging Face Spaces 步驟](#五-部署至-hugging-face-spaces-步驟)

---

## 一、 基本觀念：模型序列化

### 1. 什麼是模型部署 (Model Deployment)？
當你在 Jupyter Notebook 中訓練好一個模型後，它只存在於你電腦的記憶體中。一旦關閉程式，模型就會消失。
**模型部署**就是將模型移至雲端伺服器上長期運行。我們不能在每次用戶要預測時都重跑一遍訓練程式，這會非常耗時。

### 2. 什麼是序列化 (Serialization)？
*   **序列化**：將記憶體中的模型物件，儲存為硬碟中的二進位檔案（例如 `iris_model.joblib`）。
*   **反序列化**：當 Web 伺服器啟動時，直接讀取此二進位檔案，還原成模型物件，即可在毫秒內進行「即時預測 (Inference)」。
*   我們推薦使用 **`joblib`**，因為它對包含大量 NumPy 陣列的機器學習模型有極佳的讀寫效能。

> [!CAUTION]
> **安全性警告**：`joblib` 或 `pickle` 在反序列化時會執行任意程式碼。**千萬不要載入來源不明或未受信任的模型檔案**，否則會使伺服器面臨安全威脅！

---

## 二、 技術架構：Gradio 結合 FastAPI 的巧妙之處

### 1. 為什麼可以不用寫 Dockerfile？
如果你在 Hugging Face Spaces 上使用 **Docker SDK**，你就必須自己寫 Dockerfile 定義作業系統、Python 環境、暴露連接埠等。

但如果你選擇 **Gradio SDK**，Hugging Face 會在後台**自動幫你搞定所有的 Docker 容器化配置**。你只需要上傳程式碼，它就能自動執行。

### 2. Gradio 與 FastAPI 是如何在一起工作的？
Gradio 這個套件，底層其實是用 **FastAPI** 框架寫成的。
Gradio 提供了 `gr.mount_gradio_app()` 函數，這能讓我們做兩件事：
1.  建立一個標準的 FastAPI App，並在上面撰寫 Pydantic 的資料校驗與 `/predict` API 端點。
2.  建立 Gradio 的網頁介面。
3.  將 Gradio 網頁介面掛載到 FastAPI 的根路徑 `/` 下。

這樣一來，你的 Space 既是網頁（造訪首頁 `/`），也是 API 服務（造訪 `/predict`），而且完全不需要 Dockerfile！

---

## 三、 專案架構與程式碼說明

本範例包含以下檔案：
```text
模型部署/
├── train_save.py      # 訓練並序列化模型
├── app.py             # 結合 FastAPI 與 Gradio 的服務主程式
└── requirements.txt   # 套件依賴清單
```

### 1. 訓練模型：`train_save.py`
使用 Scikit-Learn 訓練一個簡單的鳶尾花隨機森林分類器，並使用 `joblib.dump` 將模型與類別標籤以字典格式打包儲存為 `iris_model.joblib`。

### 2. 服務主程式：`app.py`
這個檔案展示了如何將 API 與 UI 融合：
*   **自動化雙保險設計**：
    在 `app.py` 的最上方，程式會檢查當前目錄下是否存在 `iris_model.joblib`。如果沒有偵測到（例如學生部署到雲端時忘記上傳模型檔），程式會**自動呼叫 `train_save.py` 來進行線上訓練並生成模型**。這保證了服務永遠不會因為缺少模型檔而啟動失敗！
*   **FastAPI API 區塊**：
    我們定義了 Pydantic Schema（`IrisInput` 與 `IrisOutput`），限制輸入的特徵數值必須在 `0.1` 與 `10.0` 之間。如果輸入不合法，API 會自動攔截並報錯，防止模型崩潰。
*   **Gradio UI 區塊**：
    我們利用 `gr.Slider` 拉出 4 個漂亮的滑桿，並用 Markdown 格式顯示預測結果與機率百分比。
*   **融合掛載**：
    `app = gr.mount_gradio_app(app, demo, path="/")` 將兩者結合，當訪問 `/` 時顯示 Gradio 介面，訪問 `/predict` 時則是 FastAPI 預測端點。

---

## 四、 本地測試步驟

### 1. 安裝套件
請依據您使用的工具，在專案目錄下安裝所需套件：

**使用 `uv`（推薦，速度極快）：**
```bash
# 建立並啟用虛擬環境
uv venv
source .venv/bin/activate  # Windows 請用 .venv\Scripts\activate

# 安裝套件
uv pip install -r requirements.txt
```

**使用傳統 `pip`：**
```bash
# 建立並啟用虛擬環境 (以 macOS/Linux 為例)
python3 -m venv .venv
source .venv/bin/activate

# 安裝套件
pip install -r requirements.txt
```

### 2. 執行主程式
在本地，您不需要手動跑 `train_save.py`，因為 `app.py` 會自動偵測並進行初始化訓練。請直接在終端機輸入：

**使用 `uv`：**
```bash
uv run app.py
```

**使用傳統 `python`（需先啟用虛擬環境）：**
```bash
python app.py
```
看見 `INFO: Uvicorn running on http://127.0.0.1:8000` 後，代表服務已啟動。

### 3. 本地測試方式
*   **測試網頁 UI**：打開瀏覽器，造訪 `http://127.0.0.1:8000/`，您會看到一個整合了**雙分頁 (Tabs)** 的精美介面：
    1. **🔮 即時模型預測**：調整 4 個特徵滑桿，即可即時獲得預測的鳶尾花品種與對應機率。
    2. **⚙️ 線上模型訓練與評估**：調整隨機森林超參數（決策樹數量、最大深度、測試集比例、隨機種子），點擊「開始訓練模型」即可線上重訓，並動態展示最新的特徵重要性條狀圖。
*   **測試 FastAPI API (即時預測)**：
    開啟終端機，執行以下 `curl` 指令：
    ```bash
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
    ```
    您將會收到結構化的 JSON 預測回傳：
    ```json
    {
      "prediction_id": 0,
      "prediction_label": "setosa",
      "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
      }
    }
    ```

*   **測試 FastAPI API (線上重新訓練)**：
    若要線上重新訓練模型，可發送 POST 請求至 `/train` 端點，帶入超參數進行訓練：
    ```bash
    curl -X POST http://127.0.0.1:8000/train \
      -H "Content-Type: application/json" \
      -d '{"n_estimators": 50, "max_depth": 3, "test_size": 0.3, "random_state": 100}'
    ```
    您將會收到重新訓練後的評估結果與特徵重要性：
    ```json
    {
      "status": "success",
      "accuracy": 0.9556,
      "train_time": 0.0190,
      "feature_importances": {
        "sepal length": 0.0828,
        "sepal width": 0.0162,
        "petal length": 0.3973,
        "petal width": 0.5037
      },
      "message": "模型訓練完成並儲存成功！"
    }
    ```
    重訓完成後，FastAPI 後端會自動重新載入新模型，使得 `/predict` 端點與網頁 UI 均會即時套用最新的模型進行推理。

---

## 五、 部署至 Hugging Face Spaces 步驟

### 1. 設定 Hugging Face 存取憑證 (Access Token)
Hugging Face 強制要求使用 **Access Token (Write)** 作為 Git 推送的密碼。
1.  登入 Hugging Face 後，點擊右上角頭像 -> **`Settings`** -> **`Access Tokens`**。
2.  點擊 **`Create new token`**。
3.  填寫 Token 名稱（例如 `my-git-deploy`），**Token type 必須選擇 `Write`**（若選擇 Read 會無法推送程式碼）。
4.  點擊 Create 並**複製生成的 Token** 備用。

### 2. 建立 Gradio Space
1.  在 Hugging Face 右上角個人選單中點擊 **`New Space`**。
2.  填寫 Space 設定：
    *   **Space name**：例如 `iris-predict-service` (自訂名稱)
    *   **Select the Space SDK**：選擇 **`Gradio`** (注意：**千萬不要**選 Docker！)
    *   **Space hardware**：選擇免費的 **`CPU basic`**
    *   **Visibility**：**`Public`** (公開)
3.  點擊 **`Create Space`**。

### 3. 使用 Git 推送專案檔案至 Spaces
根據您的專案結構與管理習慣，我們提供以下兩種推送檔案至 Hugging Face Space 的方式：

#### 方法 A：使用 `git subtree` 直接從課程專案的子資料夾推送（推薦）
如果您的課程專案結構是一個大 Git 倉庫（例如 `machine_learning/`），而此模型部署專案位於其中的子資料夾（如 `模型部署/`），您可以使用 `git subtree` 將該子資料夾的內容作為根目錄直接推送至 Hugging Face Space，不需要額外複製檔案：

1. **確保子資料夾的變更已提交 (Commit) 到本地 Git 倉庫**：
   在您主專案的根目錄下：
   ```bash
   git add 模型部署/
   git commit -m "Commit changes before deploying to Hugging Face"
   ```

2. **使用 `git subtree push` 推送至 Hugging Face**：
   請在主專案的根目錄下執行以下指令（請替換 `你的用戶名` 與 `你的Space名稱`）：
   ```bash
   git subtree push --prefix=模型部署 https://huggingface.co/spaces/你的用戶名/你的Space名稱 main
   ```
   * **Username**：輸入您的 Hugging Face 使用者名稱。
   * **Password**：**貼上剛才申請的 Access Token (Write)**（提示：貼上密碼時畫面上不會顯示任何字元，直接貼上並按 Enter 即可）。

---

#### 方法 B：獨立複製與推送（手動複製檔案到獨立倉庫）
如果您習慣單獨為 Hugging Face Space 維護一個獨立的 Git 倉庫，可以執行以下步驟：

1. **複製 Hugging Face Space 的 Git 倉庫**：
   ```bash
   # 請替換為您的用戶名與 Space 名稱
   git clone https://huggingface.co/spaces/你的用戶名/你的Space名稱
   ```
   *執行後，您的本地會生成一個與 Space 同名的資料夾。*

2. **將專案檔案複製進去**：
   將本教學資料夾內的以下三個檔案複製到剛剛生成的 Space 資料夾下（置於根目錄）：
   *   `app.py`
   *   `train_save.py`
   *   `requirements.txt`
   *   *註：您不需要複製 `iris_model.joblib`，因為雲端啟動 `app.py` 時會自動呼叫 `train_save.py` 線上訓練！*

3. **提交並推送到 Hugging Face**：
   ```bash
   cd 你的Space名稱
   git add .
   git commit -m "Deploy Gradio + FastAPI service"
   git push
   ```
   * **Username**：輸入您的 Hugging Face 使用者名稱。
   * **Password**：**貼上剛才申請的 Access Token (Write)**（提示：貼上密碼時畫面上不會顯示任何字元，直接貼上並按 Enter 即可）。

---

### 4. 線上測試與 API 訪問
1.  推送成功後，回到 Space 網頁，狀態會從 `Building` 變成綠色的 `Running`。
2.  你的 Space 頁面上會直接顯示出漂亮的 **🌸 Iris 鳶尾花即時預測系統** 網頁介面。
3.  **如何訪問 API 端點？**
    *   你的 FastAPI 端點位置即為：`https://<你的用戶名>-<你的Space名稱>.hf.space/predict`
    *   *如何取得精確的 URL？*：
        在 Space 網頁右上角點擊三個點（`...`），選擇 **`Embed this Space`**，複製 **`Direct URL`** 的網址，並在後面接上 `/predict` 即可！
    *   現在你可以使用 Postman、Python `requests` 或 `curl` 直接對該雲端 URL 發送 POST 請求，便能獲取遠端模型的預測結果！
