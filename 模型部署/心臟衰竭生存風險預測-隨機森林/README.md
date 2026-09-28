# 🫀 心臟衰竭生存風險預測服務：FastAPI + Gradio 智慧醫療部署講義

本專案延伸自核心教學章節 [05_random_forest_heart_failure.ipynb](../../決策樹集成模型/05_random_forest_heart_failure.ipynb)，展示如何將**臨床隨機森林分類模型（Random Forest）** 完整封裝為具備 **RESTful API** 與 **Gradio 互動式 Web UI** 的現代化微服務架構。

---

## 🎯 核心學習目標

1. **多樹模型序列化**：使用 `joblib` 將 100 棵樹的隨機森林分類模型與臨床特徵字典打包序列化。
2. **Pydantic 醫療指標數值校驗**：利用 FastAPI 的 `Pydantic` 模組，對心臟射血分數 (EF)、血清肌酸酐、CPK 激酶等數值進行臨床極限值防呆驗證。
3. **急診風險分級指引**：呼叫 `predict_proba` 計算重症惡化機率，並對應急診加護病房 (ICU)、心導管檢查等處置指引。
4. **特徵重要性（Feature Importance）動態呈現**：在 Gradio 前端整合 Matplotlib 繪製水平長條圖，向醫護人員直觀呈現本次模型依賴的核心生化指標排行。
5. **模型線上熱調參**：設計 `POST /train` 端點，在線上微調樹木數量 (`n_estimators`) 與最大樹深 (`max_depth`) 並即時更新記憶體模型。

---

## 📂 專案檔案結構

```text
心臟衰竭生存風險預測-隨機森林/
├── app.py                                         # FastAPI + Gradio 混合服務（啟動主程式）
├── train_save.py                                  # 模型訓練與元數據序列化腳本
├── heart_failure_model.joblib                    # 序列化隨機森林模型檔
├── heart_failure_clinical_records_dataset.csv     # 臨床心臟衰竭數據集
├── requirements.txt                               # 相依套件清單
└── README.md                                      # 本教學講義文件
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
* **Gradio 網頁 UI**：開啟瀏覽器造訪 `http://localhost:8001`
* **Swagger API 互動文檔**：開啟瀏覽器造訪 `http://localhost:8001/docs`
