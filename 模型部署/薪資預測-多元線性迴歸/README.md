# 💼 機器學習模型部署：薪資預測-多元線性迴歸 (FastAPI + Gradio)

本專案是一個特別為機器學習教學設計的**實務部署範例**。它展示了如何將**多元線性迴歸模型**從訓練、預處理、序列化儲存、建立 RESTful API，到設計網頁互動 UI，最後一鍵部署至雲端平台（如 Render）的完整生命週期。

---

## 🎯 本課教學目標

學習完本專案，你將掌握以下核心技術：
1. **多元特徵預處理**：學會使用 `LabelEncoder` (學歷) 與 `OneHotEncoder` (城市) 對分類變數編碼，並以 `StandardScaler` 進行特徵標準化。
2. **預處理器序列化**：理解為何在部署時需要將 `LabelEncoder`、`OneHotEncoder`、`StandardScaler` 與模型一同打包序列化，以避免推理時特徵尺度與順序不一致的資料洩漏或錯誤。
3. **高可讀性 API**：使用 `FastAPI` 建立具備 `Pydantic` 強型別校驗與資料過濾機制的薪資預測端點。
4. **數學方程式動態解析**：在網頁 UI 動態將線性迴歸擬合出來的**權重係數 (Coefficients) 與截距 (Intercept)** 組裝成可讀的數學迴歸方程式，輔助學生直觀理解多元線性迴歸的數學原理。
5. **雲端網路優化**：理解反向代理（Reverse Proxy）下的緩衝與連線佇列問題，學會透過 `queue=False` 與 `.release()` 優化服務。

---

## 📂 專案架構與檔案說明

本專案目錄下包含以下核心檔案：
```text
薪資預測-多元線性迴歸/
├── app.py             # FastAPI + Gradio 融合服務（啟動主程式）
├── train_save.py      # 模型訓練與元數據序列化腳本
├── requirements.txt   # 相依套件清單
├── Salary_Data2.csv   # 薪資訓練數據集
├── ChineseFont.ttf    # 中文字型檔 (備用)
└── README.md          # 本教學講義文件
```

---

## 📖 核心觀念解析

### 1. 為什麼要保存 Preprocessors (預處理器)？
在機器學習的實際部署中，我們最常犯的錯誤是「只保存 Model 卻沒保存 Preprocessors」。
如果使用者的輸入是原始字串（如 `"大學"`, `"城市A"`），但模型需要的是經過標準化後的數值矩陣：
1. **類別對照一致性**：`LabelEncoder` 與 `OneHotEncoder` 在 `fit` 時決定的類別對應順序，必須完全與預測時一致。
2. **特徵縮放一致性**：預測時輸入的年資（例如 `5.3`）必須使用與**訓練集完全相同的均值與標準差**進行標準化。
因此，我們將 `model`、`le`、`ohe`、`scaler` 打包在同一個字典中存成 `salary_model.joblib`，保證預測與訓練的管道 (Pipeline) 完全一致。

### 2. 多元線性迴歸的數學意義
多元線性迴歸模型學得的關係可以表示為：
$$Salary = w_1 \cdot YearsExperience + w_2 \cdot EducationLevel + w_3 \cdot City\_城市A + w_4 \cdot City\_城市B + w_5 \cdot City\_城市C + b$$
*   **$w_i$ (權重係數)**：代表在其他條件不變的情況下，該特徵標準化數值增加一個單位，薪資的增減變化（萬元）。
*   **$b$ (截距)**：代表基準值。

---

## 🛠️ 本地實作指引

### 1. 環境準備與套件安裝
建議在專案目錄下使用虛擬環境執行，以避免套件衝突：
```bash
# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 啟動服務
主程式 `app.py` 設有**自動化雙保險設計**：啟動時若在目錄下找不到模型檔 `salary_model.joblib`，會自動呼叫 `train_save.py` 進行訓練並存檔，保證服務絕不因缺少模型檔而崩潰。
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
* 在 **「🔮 即時月薪預測」** 分頁中調整年資滑桿、學歷與城市下拉選單，並觀察預測薪資卡片的即時更新。
* 前往 **「⚙️ 線上模型訓練與公式解析」** 分頁，調整測試集分割比例後點擊「開始訓練」，查看決定係數 $R^2$、**迴歸方程式**與**特徵影響力長條圖**的即時變化。

#### 🅱️ 測試 FastAPI RESTful API 端點
開啟終端機，執行以下 `curl` 指令來測試我們的自訂 API 端點：

* **即時預測端點：`POST /predict`**
  ```bash
  curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"years_experience": 5.3, "education_level": "碩士以上", "city": "城市A"}'
  ```
  *回傳格式*：
  ```json
  {"predicted_salary":41.643073199564286,"estimated_annual_salary":583.0030247939}
  ```

* **線上訓練端點：`POST /train`**
  ```bash
  curl -X POST http://127.0.0.1:8000/train \
    -H "Content-Type: application/json" \
    -d '{"test_size": 0.2, "random_state": 76}'
  ```

* **API 自動文件：`/docs`**
  打開瀏覽器造訪 `http://127.0.0.1:8000/docs`，即可在 Swagger UI 進行互動式 API 測試。

---

## 🎯 雲端部署指引 (Render)

1. 將本專案推送到您的 GitHub 倉庫。
2. 登入 Render，建立一個新的 **Web Service**，並連結該 GitHub 倉庫。
3. 填寫以下設定：
   * **Root Directory**: `模型部署/薪資預測-多元線性迴歸` (若您的 GitHub 倉庫為多專案結構，請填寫此子目錄路徑；若是獨立倉庫則留空)
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `python app.py`
4. 服務啟動後，Render 會自動注入環境變數 `PORT`，程式會自動讀取並綁定。

---

## 📝 課後習題與延伸思考

1. **對照 Jupyter Notebook 驗證**：在 `Jupyter Notebook` 中最後進行了新資料預測：年資 `5.3`、學歷 `碩士以上`、城市 `城市A`，得到的預測月薪為 `41.643` 萬元。請透過你的 API (`POST /predict`) 或網頁介面輸入相同參數，驗證兩者回傳的預測薪資是否完全一致。
2. **正負影響的現實解讀**：觀察「特徵影響力」長條圖，有哪些特徵對薪資是「正向影響（加薪）」？有哪些是「負向影響（減薪）」？這是否符合你的直覺？例如「高中以下」對比基準值為什麼呈現負向影響？
3. **多元共線性**：獨熱編碼 (One-Hot Encoding) 將 `City` 轉換為三個欄位，但這在線性迴歸中可能引發「虛擬變數陷阱 (Dummy Variable Trap)」或多重共線性。如果我們在 `OneHotEncoder` 設定 `drop='first'`（只保留兩個城市欄位），模型 $R^2$ 與預測能力會有什麼變化？

---

## 授權

MIT License
