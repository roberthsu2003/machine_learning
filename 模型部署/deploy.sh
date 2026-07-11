#!/bin/bash
# deploy.sh - 乾淨地推送程式碼至 Hugging Face Space，避開二進位歷史紀錄與衝突問題

# 取得腳本所在的目錄
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=== Hugging Face 乾淨部署工具 ==="
read -p "請輸入您的 Hugging Face 用戶名 (例如: roberthsu2003): " HF_USERNAME
read -p "請輸入您的 Space 名稱 (例如: iris-predict-service): " HF_SPACE_NAME
read -sp "請貼上您的 Access Token (Write權限): " HF_TOKEN
echo ""

# 建立乾淨的臨時資料夾
DEPLOY_DIR="/tmp/hf_deploy_temp"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# 複製必要的純文字檔案，排除二進位檔案
cp "$SCRIPT_DIR/app.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/train_save.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/README.md" "$DEPLOY_DIR/"
if [ -f "$SCRIPT_DIR/.gitattributes" ]; then
    cp "$SCRIPT_DIR/.gitattributes" "$DEPLOY_DIR/"
fi
if [ -f "$SCRIPT_DIR/.gitignore" ]; then
    cp "$SCRIPT_DIR/.gitignore" "$DEPLOY_DIR/"
fi

# 初始化新倉庫並進行第一次提交
cd "$DEPLOY_DIR"
git init -b main
git config user.name "deploy"
git config user.email "deploy@example.com"
git add .
git commit -m "Deploy Gradio FastAPI service (clean history)"

# 使用 Token 自動認證並強制推送
echo "正在推送至 Hugging Face Space..."
git push "https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME" main:main --force

echo "=== 部署完成！正在清理暫存檔案 ==="
rm -rf "$DEPLOY_DIR"
echo "部署成功！您可以前往 Hugging Face Space 網頁查看狀態。"
