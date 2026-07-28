# Iris 鳶尾花機器學習服務

結合 **FastAPI** 與 **Gradio** 的機器學習部署範例,展示模型的完整生命週期:訓練、序列化、線上預測與線上重新訓練。適合作為學生學習機器學習部署的教學專案。

## 功能

- 🔮 **即時預測**:輸入鳶尾花的 4 項特徵(花萼/花瓣的長寬),即時取得品種預測與各類別機率
- ⚙️ **線上訓練**:線上調整隨機森林超參數(樹數量、最大深度、測試集比例、隨機種子),重新訓練並即時更新服務模型
- 📊 **評估與特徵重要性**:訓練後顯示測試集準確度、耗時與特徵重要性
- 🧩 **雙介面**:同一服務同時提供 Gradio Web UI 與標準 RESTful API

## 專案結構

| 檔案 | 說明 |
|---|---|
| `app.py` | FastAPI + Gradio 融合服務,啟動主程式 |
| `train_save.py` | 訓練隨機森林並將模型與元數據序列化為 `iris_model.joblib` |
| `requirements.txt` | 相依套件 |

> 首次啟動時若偵測不到 `iris_model.joblib`,`app.py` 會自動呼叫 `train_save.py` 訓練並產生模型。

## API 端點

| 方法 | 路徑 | 說明 |
|---|---|---|
| POST | `/predict` | 傳入 4 項特徵,回傳預測類別與機率分佈 |
| POST | `/train` | 傳入超參數線上重新訓練,並即時更新服務模型 |
| GET | `/docs` | Swagger UI 互動式 API 文件 |
| GET | `/openapi.json` | OpenAPI 規格 |
| GET | `/` | Gradio Web UI |

`/predict` 請求範例:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## 本地執行

```bash
pip install -r requirements.txt
python app.py
```

預設在 `http://localhost:8000` 啟動。開發時可啟用熱重載:

```bash
RELOAD=true python app.py
```

也可自訂埠號:

```bash
PORT=3000 python app.py
```

## 部署到 Render

在 Render 建立 **Web Service**,設定如下:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`(或 `uvicorn app:app --host 0.0.0.0 --port $PORT`)

服務會自動讀取 Render 注入的 `PORT` 環境變數,無需額外設定。

## 授權

MIT
