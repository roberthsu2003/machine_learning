# 🚀 部署指南

本指南將幫助您部署機器學習兩階段專案到各種平台。

## 📋 部署前準備

### 1. 確保模型已訓練
```bash
cd model_training
python train_models.py
```

### 2. 檢查生成的文件
確保 `models/` 目錄包含以下文件：
- `knn_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`
- `pytorch_model.pth`
- `scaler.pkl`
- `metadata.pkl`

## 🌐 部署選項

### 選項 1: Streamlit Cloud（推薦）

#### 步驟 1: 準備 GitHub 倉庫
1. 將專案推送到 GitHub
2. 確保所有文件都在根目錄

#### 步驟 2: 部署到 Streamlit Cloud
1. 訪問 [Streamlit Cloud](https://streamlit.io/cloud)
2. 連接您的 GitHub 帳戶
3. 選擇倉庫
4. 設定主文件路徑：`frontend/app.py`
5. 點擊 "Deploy"

#### 步驟 3: 配置環境變數（可選）
在 Streamlit Cloud 設定中添加：
```
PYTHONPATH=/app
```

### 選項 2: Heroku

#### 步驟 1: 創建必要文件

**Procfile:**
```
web: streamlit run frontend/app.py --server.port=$PORT --server.headless=true
```

**runtime.txt:**
```
python-3.9.18
```

#### 步驟 2: 部署
```bash
# 安裝 Heroku CLI
# 登入 Heroku
heroku login

# 創建應用
heroku create your-app-name

# 部署
git push heroku main
```

### 選項 3: Docker 部署

#### 本地 Docker
```bash
# 構建映像
docker build -t ml-app .

# 運行容器
docker run -p 8501:8501 ml-app
```

#### Docker Compose
```bash
# 啟動服務
docker-compose up -d

# 查看日誌
docker-compose logs -f
```

### 選項 4: Google Cloud Run

#### 步驟 1: 準備 Dockerfile
使用專案中的 Dockerfile

#### 步驟 2: 部署
```bash
# 設定專案
gcloud config set project YOUR_PROJECT_ID

# 構建並推送映像
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ml-app

# 部署到 Cloud Run
gcloud run deploy ml-app \
  --image gcr.io/YOUR_PROJECT_ID/ml-app \
  --platform managed \
  --region asia-east1 \
  --allow-unauthenticated
```

### 選項 5: AWS EC2

#### 步驟 1: 啟動 EC2 實例
- 選擇 Ubuntu 20.04 LTS
- 實例類型：t3.medium 或更大
- 安全群組：開放端口 8501

#### 步驟 2: 安裝依賴
```bash
# 更新系統
sudo apt update && sudo apt upgrade -y

# 安裝 Python 3.9
sudo apt install python3.9 python3.9-pip -y

# 安裝 Git
sudo apt install git -y

# 克隆專案
git clone YOUR_REPO_URL
cd 專案實作

# 安裝依賴
pip3 install -r requirements.txt
```

#### 步驟 3: 運行應用
```bash
# 使用 screen 或 tmux 保持會話
screen -S ml-app

# 啟動應用
streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0

# 按 Ctrl+A, D 離開 screen
```

## 🔧 配置選項

### 環境變數
```bash
# Python 路徑
export PYTHONPATH=/app

# Streamlit 配置
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### 自定義配置
編輯 `.streamlit/config.toml` 來調整：
- 主題顏色
- 端口設定
- 日誌級別

## 📊 監控和維護

### 健康檢查
應用程式提供健康檢查端點：
```
GET /_stcore/health
```

### 日誌監控
```bash
# 查看 Streamlit 日誌
streamlit run frontend/app.py --logger.level=debug

# Docker 日誌
docker logs CONTAINER_ID

# Docker Compose 日誌
docker-compose logs -f
```

### 性能優化
1. **模型快取**：模型在首次載入後會被快取
2. **數據預處理**：使用 joblib 快取預處理器
3. **並發處理**：Streamlit 支援多用戶並發

## 🚨 故障排除

### 常見問題

#### 1. 模型載入失敗
```
❌ 模型載入失敗! 請確保已運行模型訓練階段。
```
**解決方案**：
- 確保 `models/` 目錄存在且包含所有必要文件
- 檢查文件權限
- 重新運行模型訓練

#### 2. 端口被佔用
```
Port 8501 is already in use
```
**解決方案**：
```bash
# 查找佔用端口的進程
lsof -i :8501

# 終止進程
kill -9 PID

# 或使用不同端口
streamlit run frontend/app.py --server.port=8502
```

#### 3. 記憶體不足
**解決方案**：
- 增加實例記憶體
- 優化模型大小
- 使用模型量化

#### 4. 依賴安裝失敗
**解決方案**：
```bash
# 更新 pip
pip install --upgrade pip

# 使用 conda 環境
conda create -n ml-app python=3.9
conda activate ml-app
pip install -r requirements.txt
```

## 📈 擴展建議

### 1. 添加更多模型
在 `model_training/train_models.py` 中添加新的模型類型

### 2. 數據庫整合
- 使用 SQLite 存儲預測歷史
- 整合 PostgreSQL 或 MySQL

### 3. 用戶認證
- 添加 Streamlit 認證
- 整合 OAuth 登入

### 4. API 接口
- 創建 FastAPI 後端
- 提供 REST API 接口

### 5. 監控儀表板
- 整合 Prometheus 監控
- 添加 Grafana 儀表板

## 🔒 安全考慮

### 1. 環境變數
- 不要在代碼中硬編碼敏感信息
- 使用環境變數或密鑰管理服務

### 2. 網路安全
- 配置防火牆規則
- 使用 HTTPS
- 限制訪問 IP

### 3. 數據保護
- 加密敏感數據
- 定期備份模型文件
- 實施訪問控制

## 📞 支援

如果遇到問題，請：
1. 檢查本指南的故障排除部分
2. 查看 GitHub Issues
3. 聯繫專案維護者

---

**祝您部署順利！** 🎉
