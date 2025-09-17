# ⚡ 快速開始指南

本指南將幫助您在 5 分鐘內啟動機器學習兩階段專案。

## 🎯 目標

完成本指南後，您將擁有一個完整的機器學習應用程式，包含：
- 多種訓練好的分類模型
- 互動式網頁預測界面
- 模型性能比較功能

## 📋 前置要求

- Python 3.9 或更高版本
- 至少 2GB 可用記憶體
- 網路連接（用於下載依賴）

## 🚀 5 分鐘快速啟動

### 步驟 1: 安裝依賴（1 分鐘）
```bash
# 進入專案目錄
cd 專案實作

# 安裝所有依賴
pip install -r requirements.txt
```

### 步驟 2: 訓練模型（2 分鐘）
```bash
# 進入模型訓練目錄
cd model_training

# 運行模型訓練
python train_models.py
```

**預期輸出**：
```
🚀 開始機器學習模型訓練流程...
📥 載入 Iris 數據集...
✅ 數據載入完成
🤖 開始訓練傳統機器學習模型...
🧠 開始訓練 PyTorch 神經網路模型...
💾 保存模型到 ../models/...
✅ 所有模型已保存
```

### 步驟 3: 啟動前端應用（1 分鐘）
```bash
# 返回專案根目錄
cd ..

# 啟動 Streamlit 應用
streamlit run frontend/app.py
```

**預期輸出**：
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### 步驟 4: 測試應用（1 分鐘）
1. 開啟瀏覽器訪問 `http://localhost:8501`
2. 在側邊欄調整花朵測量值
3. 點擊預測按鈕
4. 查看預測結果和機率分佈

## 🎮 功能測試

### 測試 1: 基本預測
1. 使用預設值（山鳶尾範例）
2. 點擊「🔮 開始預測」
3. 確認所有模型都預測為「山鳶尾」

### 測試 2: 模型比較
1. 調整滑桿到變色鳶尾範圍：
   - 萼片長度：6.2
   - 萼片寬度：2.9
   - 花瓣長度：4.3
   - 花瓣寬度：1.3
2. 觀察不同模型的預測結果
3. 查看機率分佈圖表

### 測試 3: 模型一致性
1. 嘗試邊界值（不同模型可能給出不同預測）
2. 觀察「模型一致性分析」部分
3. 理解為什麼會出現分歧

## 🔧 故障排除

### 問題 1: 模型載入失敗
**症狀**：看到「❌ 模型載入失敗」錯誤

**解決方案**：
```bash
# 確保在正確的目錄
cd 專案實作

# 重新訓練模型
cd model_training
python train_models.py
cd ..
```

### 問題 2: 端口被佔用
**症狀**：看到「Port 8501 is already in use」

**解決方案**：
```bash
# 使用不同端口
streamlit run frontend/app.py --server.port=8502
```

### 問題 3: 依賴安裝失敗
**症狀**：pip install 失敗

**解決方案**：
```bash
# 更新 pip
pip install --upgrade pip

# 使用虛擬環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 預期結果

### 模型性能
- **KNN**: 準確率約 96-100%
- **隨機森林**: 準確率約 96-100%
- **SVM**: 準確率約 96-100%
- **PyTorch**: 準確率約 96-100%

### 界面功能
- ✅ 滑桿輸入控制
- ✅ 即時預測結果
- ✅ 機率分佈視覺化
- ✅ 模型一致性分析
- ✅ 響應式設計

## 🎓 學習要點

完成快速開始後，您應該理解：

1. **機器學習流程**：
   - 數據載入 → 預處理 → 模型訓練 → 評估 → 部署

2. **模型比較**：
   - 不同演算法的優缺點
   - 預測一致性分析
   - 機率分佈的重要性

3. **實際應用**：
   - 如何將模型整合到網頁應用
   - 用戶友好的界面設計
   - 即時預測和視覺化

## 🚀 下一步

### 進階功能
1. **添加新模型**：在 `model_training/train_models.py` 中添加更多演算法
2. **自定義界面**：修改 `frontend/app.py` 的樣式和布局
3. **數據擴展**：嘗試其他數據集（如 Wine、Breast Cancer）

### 部署選項
1. **本地部署**：使用 Docker 容器化
2. **雲端部署**：部署到 Streamlit Cloud、Heroku 等
3. **企業部署**：整合到現有系統

### 學習資源
- [Streamlit 官方文檔](https://docs.streamlit.io/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Scikit-learn 用戶指南](https://scikit-learn.org/stable/user_guide.html)

## 🆘 需要幫助？

如果遇到問題：
1. 檢查本指南的故障排除部分
2. 查看 `README.md` 詳細說明
3. 檢查 `DEPLOYMENT.md` 部署指南
4. 聯繫專案維護者

---

**恭喜！您已成功啟動機器學習應用程式！** 🎉
