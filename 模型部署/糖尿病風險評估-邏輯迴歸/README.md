# 🩺 糖尿病患病風險評估服務：FastAPI + Gradio 智慧醫療部署講義

本專案延伸自核心教學章節 [03_diabetes_logistic_regression.ipynb](../../邏輯迴歸/03_diabetes_logistic_regression.ipynb)，展示如何將**臨床邏輯迴歸分類模型（Logistic Regression）** 完整封裝為具備 **RESTful API** 與 **Gradio 互動式 Web UI** 的現代化微服務架構。

---

## 🎯 核心學習目標

1. **模型與預處理器聯合序列化**：使用 `joblib` 將訓練好的邏輯迴歸分類模型與 `StandardScaler` 標準化器聯合儲存，確保推論階段資料尺度的一致性。
2. **Pydantic 資料檢驗**：利用 FastAPI 的 `Pydantic` 模組，對病患生化檢驗數值（年齡、體重、空腹血糖）進行嚴格的生理合理範圍驗證。
3. **臨床機率輸出與分級**：呼叫 `predict_proba` 計算患病機率（$0\% \sim 100\%$），並對應臨床「低風險、中度風險、極高風險」處置指引。
4. **危險因子勝算比（Odds Ratio）即時圖表**：在 Gradio 前端整合 Matplotlib 動態長條圖，解析血糖與年齡等危險因子的風險加權。
5. **模型線上熱重載（Hot Reloading）**：設計 `POST /train` 端點，在線上動態微調正則化強度 $C$ 並即時更新記憶體模型。

---

## 📂 專案檔案結構

```text
糖尿病風險評估-邏輯迴歸/
├── app.py                     # FastAPI + Gradio 混合服務（啟動主程式）
├── train_save.py              # 模型訓練、標準化與元數據序列化腳本
├── diabetes_model.joblib      # 序列化二進位模型檔（包含 scaler 與 metadata）
├── Diabetes_Data.csv          # 臨床糖尿病數據集
├── requirements.txt           # 相依套件清單
└── README.md                  # 本教學講義文件
```

---

## 🚀 快速啟動指南

### 1. 安裝相依套件
```bash
pip install -r requirements.txt
```

### 2. 訓練並生成模型檔
```bash
python train_save.py
```

### 3. 啟動服務
```bash
python app.py
```
* **Gradio 網頁 UI**：開啟瀏覽器造訪 `http://localhost:8000`
* **Swagger API 互動文檔**：開啟瀏覽器造訪 `http://localhost:8000/docs`
