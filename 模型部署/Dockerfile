# 使用官方輕量級 Python 映像檔
FROM python:3.10-slim

# 設定環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 設定工作目錄
WORKDIR /code

# 複製依賴項目清單並安裝
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 建立一個非 root 用戶以符合 Hugging Face 安全規範
# Hugging Face Spaces 預設會使用 UID 1000 執行容器
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 設定用戶工作目錄
WORKDIR $HOME/app

# 複製當前目錄下所有檔案至容器中，並將擁有者改為 user
COPY --chown=user . $HOME/app

# 在建置階段自動訓練並生成模型檔案 (iris_model.joblib)
# 這樣可以確保 Docker 映像檔中一定包含最新的模型，避免啟動時找不到檔案
RUN python train_save.py

# 暴露 7860 連接埠（Hugging Face Spaces 規定必須使用 7860 端口）
EXPOSE 7860

# 啟動 FastAPI 服務
# 注意：必須設定為 0.0.0.0，才能讓容器外訪問
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
