#!/bin/bash
# deploy.sh - 鳶尾花隨機森林分類器 Render 部署輔助工具

# 取得腳本所在的目錄
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEBHOOK_FILE="$SCRIPT_DIR/.render_webhook"

echo "============================================="
echo "        Render 部署輔助與自動化觸發工具        "
echo "============================================="
echo ""
echo "本專案支援兩種 Render 部署方式："
echo "1. GitHub 自動部署（推薦）："
echo "   將程式碼推送到 GitHub 後，Render 會自動偵測並重新建置部署。"
echo "2. Deploy Webhook 手動觸發："
echo "   當您關閉了 Render 的 Auto Deploy，或是想要手動即時觸發更新時，"
echo "   可以使用此腳本發送 Deploy Hook 請求。"
echo ""
echo "---------------------------------------------"
echo "步驟 1：請確保您的最新程式碼已提交並推送到 GitHub"
echo "        git add ."
echo "        git commit -m 'Update configuration for Render'"
echo "        git push"
echo "---------------------------------------------"
echo ""

# 檢查是否有儲存過的 Webhook URL
SAVED_URL=""
if [ -f "$WEBHOOK_FILE" ]; then
    SAVED_URL=$(cat "$WEBHOOK_FILE")
fi

HOOK_URL=""
if [ -n "$SAVED_URL" ]; then
    echo "偵測到先前儲存的 Render Deploy Hook URL："
    echo "  $SAVED_URL"
    read -p "是否使用此 URL 觸發部署？ [Y/n]: " USE_SAVED
    USE_SAVED=${USE_SAVED:-Y}
    if [[ "$USE_SAVED" =~ ^[Yy]$ ]]; then
        HOOK_URL="$SAVED_URL"
    fi
fi

if [ -z "$HOOK_URL" ]; then
    echo "請輸入您的 Render Deploy Hook URL"
    echo "(可在 Render 網頁後台 -> 您的 Web Service -> 設定頁面中找到 Deploy Hook 欄位)："
    read -p "URL: " HOOK_URL
    echo ""
fi

if [ -n "$HOOK_URL" ]; then
    # 儲存 Webhook URL 以備下次使用
    echo "$HOOK_URL" > "$WEBHOOK_FILE"
    
    echo "正在發送部署觸發請求至 Render..."
    RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$HOOK_URL")
    
    # 取得 HTTP 狀態碼
    HTTP_STATUS=$(echo "$RESPONSE" | tr -d '\r' | grep "HTTP_STATUS" | cut -d':' -f2)
    # 取得回應內容
    BODY=$(echo "$RESPONSE" | grep -v "HTTP_STATUS")
    
    if [ "$HTTP_STATUS" -eq 200 ] || [ "$HTTP_STATUS" -eq 201 ] || [ "$HTTP_STATUS" -eq 204 ]; then
        echo "✅ 部署請求已成功發送！Render 正在開始拉取最新代碼並進行建置。"
        echo "您可以前往 Render Dashboard 查看即時建置日誌。"
    else
        echo "❌ 部署觸發失敗！"
        echo "HTTP 狀態碼: $HTTP_STATUS"
        echo "回應內容: $BODY"
    fi
else
    echo "未提供 Deploy Hook URL，已跳過自動觸發。"
    echo "請手動至 Render 控制台或等待 GitHub 自動部署。"
fi

echo ""
echo "============================================="
