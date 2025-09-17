# 🤖 機器學習兩階段專案實作

這是一個完整的機器學習專案，分為兩個階段：**模型訓練**和**前端展示**。

## 📁 專案結構

```
專案實作/
├── model_training/          # 第一階段：模型訓練
│   ├── train_models.py      # 主訓練腳本
│   ├── models.py           # PyTorch 模型定義
│   ├── data_loader.py      # 數據載入和預處理
│   ├── requirements.txt    # 訓練階段依賴
│   └── README.md          # 訓練階段說明
├── frontend/               # 第二階段：前端展示
│   ├── app.py             # Streamlit 主應用
│   ├── model_loader.py    # 模型載入器
│   ├── requirements.txt   # 前端依賴
│   └── README.md         # 前端說明
├── models/                # 訓練好的模型存儲
├── data/                  # 數據文件
├── requirements.txt       # 總體依賴
└── README.md             # 本文件
```

## 🚀 快速開始

### 第一階段：模型訓練
```bash
cd model_training
pip install -r requirements.txt
python train_models.py
```

### 第二階段：前端展示
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 🎯 功能特色

### 模型訓練階段
- **多種演算法**：KNN、隨機森林、SVM、PyTorch 神經網路
- **數據預處理**：標準化、數據分割
- **模型評估**：準確率、混淆矩陣、分類報告
- **模型保存**：支援多種格式保存

### 前端展示階段
- **互動式預測**：即時輸入和預測
- **多模型比較**：同時使用多個模型預測
- **視覺化展示**：機率分佈、性能比較
- **響應式設計**：適配不同螢幕尺寸

## 📚 學習目標

完成本專案後，您將掌握：
- 機器學習完整流程
- PyTorch 深度學習框架
- Streamlit 網頁應用開發
- 模型部署和展示技巧
