# 機器學習模型部署：免 Dockerfile 的 FastAPI + Gradio 混合服務部署教學

在機器學習專案中，當我們訓練好模型後，通常會面臨兩個開發需求：
1.  **建立 Web API (例如 FastAPI)**：提供一個高效、規格化的預測端點，讓外部程式（如 App 或其他伺服器）能透過發送 JSON 請求來獲取預測結果。
2.  **建立 網頁 UI 介面 (例如 Gradio)**：提供一個簡單、直觀的網頁，讓非程式背景的客戶或同學可以直接在瀏覽器上操作（如調整滑桿、按按鈕）並即時看到結果。

以往要部署這兩套系統，可能需要撰寫複雜的 **Dockerfile** 將其容器化。**本教學將介紹一個極為巧妙的「免 Dockerfile」方案**：利用 Gradio 底層本來就是 FastAPI 的特性，在 Gradio 的 Space 中直接掛載 FastAPI 端點。

這樣一來，學生**既能完整學習到 FastAPI 與 Pydantic 的 API 開發，又不需要學習 Docker 觀念**，還能同時擁有一套免費且漂亮的網頁介面！

---

<a id="目錄"></a>
## 目錄
1. [基本觀念：模型序列化](#一-基本觀念模型序列化)
2. [技術架構：Gradio 結合 FastAPI 的巧妙之處](#二-技術架構gradio-結合-fastapi-的巧妙之處)
3. [專案架構與程式碼說明](#三-專案架構與程式碼說明)
4. [本地測試步驟](#四-本地測試步驟)
5. [部署至 Render 步驟](#五-部署至-render-步驟)

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

[↩️ 返回目錄](#目錄)

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

[↩️ 返回目錄](#目錄)

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
    我們定義了 Pydantic Schema（`IrisInput` 與 `IrisOutput`），限制輸入的特徵數值必須在 `0.1` 與 `10.0` 之間�## 五、 部署至 Render 步驟

### 1. 前置作業
Render 主要是透過與您的 **GitHub** 連動來進行自動部署。因此在開始部署前，請先將專案程式碼上傳至您的 GitHub 儲存庫：
```bash
git add .
git commit -m "Configure service for Render deployment"
git push origin main
```

> [!NOTE]
> **本專案的自動化設計：**
> 1. 我們在專案根目錄下附帶了 `render.yaml` 配置文件，支援 Render Blueprint 一鍵部署。
> 2. 雲端建置時，Render 會自動透過 `requirements.txt` 安裝相依套件，並執行 `train_save.py` 初始化訓練模型，確保服務啟動前即具備可用的模型檔案。

---

### 2. 建立與部署 Web Service

您可以使用以下兩種方式在 Render 上部署本專案：

#### 方法 A：使用 Blueprint 宣告式部署（推薦 🚀）
這是一種最為方便的「基礎架構即程式碼 (IaC)」部署方式，能自動載入專案的 `render.yaml` 進行設定：
1. 登入 [Render Dashboard](https://dashboard.render.com/)。
2. 點擊右上角 **`New +`** ➡️ 選擇 **`Blueprint`**。
3. 連接包含本專案的 GitHub 儲存庫。
4. 填寫 Service Group 名稱，Render 會自動解析根目錄的 `render.yaml` 並帶出配置項目。
5. 點擊 **`Approve`**，Render 就會自動建立 Web Service 並開始拉取程式碼建置。

#### 方法 B：手動在 Render 控制台建立服務
如果您想手動微調配置，也可以透過控制台按鈕逐步設定：
1. 在 Render Dashboard 點擊右上角 **`New +`** ➡️ 選擇 **`Web Service`**。
2. 選擇 **`Build and deploy from a Git repository`**，並連接您的 GitHub 儲存庫。
3. 填寫以下設定：
   * **Name**：`iris-predict-service` (可自訂)
   * **Region**：選擇距離您較近的區域 (例如 `Singapore`)
   * **Branch**：`main` (或您程式碼所在的分支)
   * **Language**：`Python`
   * **Build Command**：`pip install -r requirements.txt && python train_save.py`
   * **Start Command**：`uvicorn app:app --host 0.0.0.0 --port $PORT`
   * **Instance Type**：選擇 **`Free`** (免費層)
4. 點擊 **`Create Web Service`** 即可開始部署。

> [!WARNING]
> **免費層實例休眠限制：**
> Render 的免費層服務在 **15 分鐘無流量** 後會自動進入休眠狀態。當下一個新請求進來時，服務會被重新喚醒，這時可能需要 **50 秒左右** 的冷啟動時間，請耐心等候。

---

### 3. 使用 `deploy.sh` 觸發手動部署
當您在 Render 上關閉了 `Auto Deploy`（自動部署），或是想要在本地推送代碼至 GitHub 後即時手動更新時，可以使用專案附帶的 [deploy.sh](file:///Users/roberthsu2003/Documents/GitHub/machine_learning/模型部署/deploy.sh) 腳本。

1. 在您的 Render Web Service 控制台 ➡️ 進入 **`Settings`** 分頁。
2. 找到 **`Deploy Hook`** 欄位並複製其 URL。
3. 在本地專案終端機執行：
   ```bash
   ./deploy.sh
   ```
4. 依提示貼上該 Deploy Hook URL，腳本會自動發送 POST 請求通知 Render 拉取最新程式碼重建服務。

---

### 4. 線上測試與 API 訪問說明

當 Render 顯示部署狀態為綠色的 `Live` 後，您可以取得專案的專屬網址（例如：`https://iris-predict-service.onrender.com`）。

1. **訪問網頁 UI (Gradio)**：
   直接在瀏覽器打開您的服務網址：`https://<您的服務名稱>.onrender.com/`。即可體驗與本地相同的預測與線上重訓功能。

2. **訪問互動式 API 文件 (Swagger UI)**：
   造訪 `https://<您的服務名稱>.onrender.com/docs`。即可在此查看並直接測試所有開放的 FastAPI 端點。

3. **使用 curl 進行 API 測試**：
   * **🔮 即時預測端點：`POST /predict`**
     ```bash
     curl -X POST https://<您的服務名稱>.onrender.com/predict \
       -H "Content-Type: application/json" \
       -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
     ```
   * **⚙️ 線上重訓端點：`POST /train`**
     ```bash
     curl -X POST https://<您的服務名稱>.onrender.com/train \
       -H "Content-Type: application/json" \
       -d '{"n_estimators": 50, "max_depth": 3, "test_size": 0.3, "random_state": 100}'
     ```

[↩️ 返回目錄](#目錄)�� -b temp-deploy

   # (2) 將此臨時分支強制推送至 Hugging Face 的 main 分支
   git push https://huggingface.co/spaces/你的用戶名/你的Space名稱 temp-deploy:main --force

   # (3) 刪除本地臨時分支以保持乾淨
   git branch -D temp-deploy
   ```

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

### ⚠️ 常見錯誤：Git 推送被拒絕 (rejected)

當您執行 `git subtree push` 或 `git push` 時，可能會遇到類似以下的錯誤：
```text
 ! [rejected]        b1093d0d923a863e35fa64c390311c04d3af326b -> main (fetch first)
錯誤: 推送一些引用到 'https://huggingface.co/spaces/你的用戶名/你的Space名稱' 失敗
提示： 更新被拒絕，因為遠端包含您本機沒有的提交。這通常是因為另一個版本庫有推送更動...
```

#### 1. 為什麼會這樣？
當您在 Hugging Face 網站上建立新 Space 時，系統會自動在該遠端倉庫中建立初始檔案（例如包含 Space 設定的 `README.md` 元數據與 `.gitattributes`）。這些提交存在於遠端，但您本地子資料夾中沒有這些提交，因此 Git 為了防止覆蓋而拒絕了推送。

> [!CAUTION]
> **重要警告**：Hugging Face 必須依賴倉庫根目錄下的 `README.md` 最頂部的 YAML 設定區塊（定義了 `title`、`sdk: gradio` 等）來啟動服務。**請勿**在強制推送時將其抹除，否則 Space 會因為找不到 SDK 設定而建置失敗。

#### 2. 如何解決？
我們提供以下三種解決方案，最推薦使用 **解決方案一（部署腳本）**，它會自動為您排除所有二進位檔案與 Git 歷史衝突問題，免去複雜的手動指令：

* **解決方案一：使用一鍵部署腳本 `deploy.sh`（終極推薦 ⚡️）**
  我們在專案中為您撰寫了一個 [deploy.sh](file:///Users/roberthsu2003/Documents/GitHub/machine_learning/模型部署/deploy.sh) 部署腳本。它會自動將純文字代碼檔案（排除 `iris_model.joblib` 二進位檔）提取至系統暫存區，建立乾淨的全新 Git 歷史紀錄，並強制推送到 Hugging Face Space：
  
  1. 請在終端機中切換至 `模型部署/`（或 `test/`）子目錄：
     ```bash
     cd 模型部署
     ```
  2. 執行部署腳本：
     ```bash
     ./deploy.sh
     ```
  3. 依提示輸入您的 Hugging Face 用戶名、Space 名稱以及 Access Token (Write 權限)，腳本將自動為您完成乾淨的部署，完成後會自動清空暫存檔案！

---

* **解決方案二：克隆並複製設定檔至本地子資料夾，再使用 `git subtree` 強制推送**
  1. **將 Hugging Face Space 克隆至主專案「外部」的暫存資料夾**：
     ```bash
     # 請克隆至與您主專案資料夾平級的外層目錄（例如 Documents/GitHub/ 下）
     git clone https://huggingface.co/spaces/你的用戶名/你的Space名稱
     ```
  2. **將設定檔複製回您主專案的子資料夾**：
     將克隆下來的 Space 資料夾底下的 `README.md`（包含最頂部的 `---` 區塊）和 `.gitattributes` 複製並覆蓋到您主專案的子資料夾（如 `模型部署/`）下。
  3. **刪除該外部暫存資料夾**：
     複製完成後，將剛才在外面 clone 的暫存 Space 資料夾刪除。
  4. **提交變更至您的主專案**：
     在主專案根目錄下：
     ```bash
     git add 模型部署/
     git commit -m "Add Hugging Face config files to subfolder"
     ```
  5. **使用臨時分支強制推送**：
     ```bash
     # 1. 將子資料夾（以「模型部署」為例）分割成一個臨時分支 temp-deploy
     git subtree split --prefix=模型部署 -b temp-deploy

     # 2. 強制推送至 Hugging Face 的 main 分支
     git push https://huggingface.co/spaces/你的用戶名/你的Space名稱 temp-deploy:main --force

     # 3. 刪除本地臨時分支
     git branch -D temp-deploy
     ```

---

* **解決方案三：獨立倉庫複製與手動推送**
  1. 將 Hugging Face Space 克隆至主專案**「外部」**的獨立資料夾（例如與您的課程主專案資料夾平級的目錄。**切勿** clone 在主專案內部，以防巢狀 Git 倉庫衝突）。
  2. 將您的代碼檔案（`app.py`、`requirements.txt` 等）複製到該獨立資料夾中（**保留** `README.md` 最上方的 `---` 設定區塊，且**不要**複製 `iris_model.joblib`）。
  3. 進入該獨立資料夾，執行標準的 `git add .`、`git commit` 與 `git push` 推送。

---

### 4. 線上測試、獨立網址與 API 文件訪問

1. **Space 狀態確認**：
   推送成功後，回到 Space 網頁，狀態會從 `Building` 轉為綠色的 `Running`。

2. **如何取得獨立網址 (Direct URL)**：
   若要跳過 Hugging Face 的外殼（Iframe），直接打開應用的網頁或 API：
   * **方式一（網址格式）**：獨立網址格式為 `https://<你的用戶名>-<你的Space名稱>.hf.space`（注意：用戶名與 Space 名稱之間的連接號是減號 `-`）。
   * **方式二（介面複製）**：在 Space 網頁右上角點擊 **三個點 `...`** ➡️ 選擇 **`Embed this Space`** ➡️ 複製 **`Direct URL`** 欄位的網址。

3. **如何訪問與測試 Swagger API 文件**：
   由於我們在程式中手動繞過了 Gradio 的路徑劫持，您現在可以直接訪問獨立網址的 `/docs` 路徑來打開 FastAPI 的互動式 API 測試文件：
   * **文件網址**：`https://<你的用戶名>-<你的Space名稱>.hf.space/docs`
   * **測試方式**：在網頁上展開端點，點擊 `Try it out`，輸入預測特徵的 JSON，即可直接發送測試。

4. **我們自訂的 API 端點說明與測試指令**：
   Swagger UI 中會包含 Gradio 本身的系統端點與我們自訂的機器學習服務端點。我們自訂的 API 如下：

   * **🔮 即時預測端點：`POST /predict`**
     * **用途**：傳入 4 個鳶尾花測量數值，模型會即時返回品種預測與機率分布。
     * **測試指令 (curl)**：
       ```bash
       curl -X POST https://<你的用戶名>-<你的Space名稱>.hf.space/predict \
         -H "Content-Type: application/json" \
         -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
       ```

   * **⚙️ 線上重訓端點：`POST /train`**
     * **用途**：傳入新的超參數重新訓練模型，重新訓練成功後，服務會自動套用新模型。
     * **測試指令 (curl)**：
       ```bash
       curl -X POST https://<你的用戶名>-<你的Space名稱>.hf.space/train \
         -H "Content-Type: application/json" \
         -d '{"n_estimators": 50, "max_depth": 3, "test_size": 0.3, "random_state": 100}'
       ```

[↩️ 返回目錄](#目錄)
