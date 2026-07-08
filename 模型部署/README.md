# 機器學習模型部署：Scikit-Learn + FastAPI + Hugging Face Spaces 教學

本教學將引導你了解如何將一個訓練好的機器學習模型轉化為生產環境中的實用 Web 服務。我們將從「模型序列化」開始，使用 **FastAPI** 建立 API 服務，最後將其容器化（Docker）並部署到免費的雲端平台 **Hugging Face Spaces**。

---

## 目錄
1. [機器學習模型部署的基本觀念](#一-機器學習模型部署的基本觀念)
2. [FastAPI 的基本觀念](#二-fastapi-的基本觀念)
3. [專案架構與程式碼說明](#三-專案架構與程式碼說明)
4. [本地測試步驟](#四-本地測試步驟)
5. [部署至 Hugging Face Spaces 步驟](#五-部署至-hugging-face-spaces-步驟)

---

## 一、 機器學習模型部署的基本觀念

### 1. 什麼是模型部署 (Model Deployment)？
在機器學習的開發流程中，我們通常會在 Jupyter Notebook 等工具中載入數據、訓練模型並進行評估。然而，一個「留在電腦筆記本裡」的模型是無法直接為使用者提供價值的。

**模型部署**就是將訓練好的模型放到伺服器上運行，使其可以接收來自外部（如網頁、App、其他伺服器）的輸入數據，並即時回傳預測結果的過程。

### 2. 為什麼要進行「模型序列化」？
*   **避免重複訓練**：機器學習模型的訓練過程可能需要數小時、數天甚至數週。我們不能在每次用戶想要預測時，都重新跑一次訓練程式。
*   **序列化 (Serialization)**：將記憶體中的 Python 對象（例如訓練好的模型結構與權重）轉換成可以儲存到硬碟的二進位檔案（如 `.joblib` 或 `.pkl`）。
*   **反序列化 (Deserialization)**：當 Web 伺服器啟動時，直接從硬碟讀取這個二進位檔案，還原成記憶體中的模型對象，即可在微秒級的時間內進行「預測推論 (Inference)」。

### 3. 常見的序列化工具：`pickle` 與 `joblib`
*   **`pickle`**：Python 的內建模組，可序列化幾乎所有的 Python 對象。
*   **`joblib`**：Scikit-Learn 官方推薦的工具。它特別針對包含大量 NumPy 大陣列的對象進行優化，讀寫大型 ML 模型時速度更快、更節省記憶體。
*   > [!CAUTION]
> **安全性警告**：`pickle` 與 `joblib` 在載入未知來源的模型檔案時具有安全風險，因為它們在反序列化時會執行任意程式碼。**切勿載入來源不明或未受信任的模型檔案！**

---

## 二、 FastAPI 的基本觀念

當模型準備好後，我們需要一個「介面」讓外部能與它溝通。這個介面通常就是 **Web API (Application Programming Interface)**。

### 1. 為什麼選擇 FastAPI？
FastAPI 是目前 Python 最熱門的 Web 框架之一，非常適合用來部署機器學習模型，其主要優勢包括：
*   **極高的效能**：基於 Starlette 與 Pydantic，效能與 Node.js 和 Go 相當。
*   **異步編程支持 (Async/Await)**：能高效率處理大量並行請求。
*   **自動生成互動式文件**：服務啟動後，造訪 `/docs` 即可自動生成 **Swagger UI**，方便開發者直接在瀏覽器測試 API 接口。
*   **基於 Pydantic 的資料校驗**：使用標準的 Python 型別標註來規範輸入與輸出格式。

### 2. 資料校驗 (Data Validation) 對 ML 的重要性
機器學習模型非常脆弱。如果模型需要 4 個 float 數值作為輸入，而用戶卻傳送了字串、`None`、或者只傳送了 3 個數值，模型在推論時就會直接崩潰（報錯）甚至給出荒謬的預測。

FastAPI 透過 **Pydantic** 來強制進行資料型別與數值範圍的校驗。例如，我們可以限制「花瓣長度」必須是 `0.1` 到 `10.0` 之間的正浮點數。如果用戶輸入了非法資料，FastAPI 會在資料傳入模型前攔截，自動回傳 `422 Unprocessable Entity` 的錯誤訊息與原因，從而保護了後端模型的穩定性。

---

## 三、 專案架構與程式碼說明

本範例包含以下檔案：
*   `train_save.py`：訓練隨機森林模型並儲存。
*   `main.py`：FastAPI 主程式。
*   `requirements.txt`：列出依賴的 Python 套件。
*   `Dockerfile`：用於容器化部署。

```text
模型部署/
├── train_save.py      # 訓練並序列化模型
├── main.py            # FastAPI API 端點程式
├── requirements.txt   # 套件依賴清單
└── Dockerfile         # Docker 映像檔設定檔
```

### 1. 訓練與儲存模型：`train_save.py`
我們使用著名的 **Iris (鳶尾花) 數據集**，以隨機森林分類器訓練模型。
*   除了模型本身，我們還將類別名稱（`target_names`，如 `['setosa', 'versicolor', 'virginica']`）打包成字典一起儲存。這樣在 API 中就能直接回傳人類看得懂的標籤名稱，而不是只有冷冰冰的數字 `0, 1, 2`。

### 2. FastAPI 服務設計：`main.py`
在 `main.py` 中，我們運用了以下設計：
*   **Lifespan 事件管理**：
    我們使用 `@asynccontextmanager` 來管理 FastAPI 的生命週期。在服務「啟動時」載入 `iris_model.joblib`，並將其暫存在 `app.state.model` 中。這樣做的好處是：
    1. 避免每次有用戶呼叫預測端點時都重新從硬碟讀取模型檔案（這會導致 API 極慢）。
    2. 如果模型檔案不存在或毀損，服務會在啟動初期立即崩潰退出，避免帶病上線。
*   **定義 Schema**：
    *   `IrisInput` (輸入限制)：限制輸入必須為 4 個特徵值，且每個數值需介於 `0.1` 與 `10.0` 之間。
    *   `IrisOutput` (輸出規範)：除了預測出來的類別 ID 與名稱，還包含每個類別的預測機率分布（例如 `{"setosa": 0.95, "versicolor": 0.05, ...}`），這對決策非常有用。
*   **預測端點 `/predict`**：
    將接收到的 Pydantic 資料轉換成二維陣列形式（如 `[[5.1, 3.5, 1.4, 0.2]]`），丟入 Scikit-Learn 模型進行 `predict()` 與 `predict_proba()`，最後將結果封裝回傳。

---

## 四、 本地測試步驟

請依照以下步驟在你的電腦上測試這套服務：

### 1. 安裝套件
確保你在虛擬環境下，並安裝本教學所需的套件：
```bash
pip install -r requirements.txt
```

### 2. 訓練並生成模型
執行訓練腳本，這會在目前的資料夾下生成 `iris_model.joblib` 檔案：
```bash
python train_save.py
```

### 3. 啟動 FastAPI 服務
使用 Uvicorn 啟動本地開發伺服器：
```bash
uvicorn main:app --reload
```
看到終端機顯示 `INFO: Uvicorn running on http://127.0.0.1:8000` 即代表啟動成功。

### 4. 測試 API
1.  打開瀏覽器，造訪健康檢查端點：`http://127.0.0.1:8000/`
2.  造訪自動文件：`http://127.0.0.1:8000/docs`
3.  在 Swagger UI 中點擊 **`POST /predict`** -> **`Try it out`** -> 修改 JSON 數據 -> 點擊 **`Execute`**。你將會看到模型返回的預測結果。

---

## 五、 部署至 Hugging Face Spaces 步驟

**Hugging Face Spaces** 是一個免費的平台，非常適合託管機器學習 Demo。

### 1. 什麼是 Docker 部署？
如果我們直接將 Python 程式碼上傳到雲端，可能會因為雲端伺服器的作業系統版本、Python 版本、甚至是底層庫的差異，導致程式無法運行。

**Docker** 是一種容器化技術。我們撰寫一個 `Dockerfile` 設定檔，它就像一張「藍圖」，定義了這個服務運作所需的一切：
*   使用什麼作業系統與 Python 版本 (`python:3.10-slim`)
*   需要安裝哪些 Python 套件 (`pip install -r requirements.txt`)
*   要複製哪些程式碼與模型檔案到容器中。
*   要暴露哪一個連接埠，以及啟動容器時要執行什麼指令。

藉由 Docker，我們能保證「在本地能動，在 Hugging Face Spaces 上也絕對能動」。

### 2. Hugging Face Spaces 的 Docker 規範
1.  **監聽埠限制**：Hugging Face規定，容器內部的 API 服務必須監聽 **`7860`** 端口，且 IP 必須設為 `0.0.0.0`。
2.  **非 Root 權限**：為了安全起見，Hugging Face 不允許容器以 `root`（系統管理員）身分執行。因此，我們的 `Dockerfile` 建立了 UID 為 `1000` 的 `user` 用戶來運行程式。

### 3. 逐步部署教學

#### 步驟 1：建立 Hugging Face 帳號與 Space
1.  前往 [Hugging Face 官網](https://huggingface.co/) 註冊並登入。
2.  點擊右上角的個人頭像，選擇 **`New Space`**。
3.  填寫 Space 的設定：
    *   **Space name**：例如 `iris-fastapi-service` (自訂名稱)
    *   **License**：例如 `mit`
    *   **Select the Space SDK**：選擇 **`Docker`**
    *   **Docker template**：選擇 **`Blank`** (空白範本)
    *   **Space hardware**：選擇免費的 **`CPU basic`**
    *   **Visibility**：**`Public`** (公開)
4.  點擊下方的 **`Create Space`**。

#### 步驟 2：使用 Git 部署專案檔案

使用 Git 部署是開發流程中推薦的做法。Hugging Face Spaces 提供了專屬的 Git 儲存庫，我們可以將其視為另一個 Git 遠端倉庫來推送程式碼。

##### 1. 設定 Hugging Face 存取憑證 (Access Token)
Hugging Face 不支援使用傳統密碼進行 Git 推送。在執行 Git 操作前，你需要生成一個 **Access Token** 作為你的推送密碼：
1.  登入 Hugging Face 後，造訪設定頁面：點擊頭像 -> **`Settings`** -> **`Access Tokens`**。
2.  點擊 **`Create new token`**。
3.  填寫 Token 名稱（例如 `git-deploy`），**Token type 必須選擇 `Write`**（寫入權限，若選 Read 會無法 Push）。
4.  點擊 Create，然後**複製生成的 Token**（該 Token 只會顯示一次，請先存放在安全的地方）。

##### 2. 使用 Git 命令列將專案推送至 Space
開啟終端機 (Terminal)，切換到你想存放 Space 專案的地方，執行以下步驟：

1.  **複製 Hugging Face Space 的 Git 儲存庫**：
    ```bash
    # 請替換為你的用戶名與 Space 名稱
    git clone https://huggingface.co/spaces/你的用戶名/你的Space名稱
    ```
    *執行後，會在你當前路徑下生成一個與 Space 同名的資料夾。*

2.  **將範例程式碼複製到該 Space 資料夾下**：
    將本教學資料夾內的以下四個檔案複製到剛才 clone 下來的資料夾中：
    *   `main.py`
    *   `train_save.py`
    *   `requirements.txt`
    *   `Dockerfile`
    *   > [!NOTE]
        > **不需要複製 `iris_model.joblib` 檔案本身**。因為我們的 `Dockerfile` 中有一行 `RUN python train_save.py`，當你將程式碼推送至雲端後，Hugging Face 會在建置環境時，自動在雲端執行該訓練程式並生成模型。這不僅避免了用 Git 傳送大型模型二進位檔的麻煩，也保證了模型程式碼的可重現性。

3.  **提交並推送到 Hugging Face**：
    進入該資料夾，執行 Git 提交與推送：
    ```bash
    cd 你的Space名稱
    git add .
    git commit -m "Deploy FastAPI service to HF Spaces"
    git push
    ```
    *當終端機提示要求輸入使用者名稱與密碼時：*
    *   **Username**：輸入你的 Hugging Face 使用者名稱。
    *   **Password**：**貼上你剛才複製的 Hugging Face Access Token (Write)**。（提示：貼上密碼時畫面上不會顯示任何字元，這是正常的，直接貼上並按 Enter 即可）。

#### 步驟 3：等待建置與測試
1.  上傳完成後，點擊 Space 頂部的 **`App`** 標籤。
2.  你會看到狀態顯示為 `Building`，此時 Hugging Face 正在依照你的 `Dockerfile` 下載環境、安裝套件並訓練模型。
3.  建置完成後，狀態會變成綠色的 `Running`。
4.  由於我們部署的是純 API 服務，Hugging Face 的網頁展示框可能會顯示「沒有網頁界面」或顯示我們的健康檢查 JSON。
5.  **如何訪問 API 文件？**
    你的 API 文件網址為：
    `https://huggingface.co/spaces/你的用戶名/你的Space名稱/resolve/main/` 的應用網址。
    最快查看運行 API 的方式是：
    點擊 Space 頁面右上角的三個點（`...`），選擇 **`Embed this Space`**，在跳出的視窗中找到 **`Direct URL`**，例如 `https://username-space-name.hf.space`。
    將該網址複製出來，並在後面加上 **`/docs`** (例如 `https://username-space-name.hf.space/docs`)。你就可以在瀏覽器中看到你的線上 Swagger UI，並直接發送測試請求！

---

## 結語與課後練習
恭喜你！你已經學會了將機器學習模型轉為線上服務的完整流程。
*   **思考題**：如果我們要更新模型（例如增加特徵，或更換為 XGBoost 演算法），我們的 `main.py` 與 `Pydantic` 輸入需要做出什麼對應的調整？
