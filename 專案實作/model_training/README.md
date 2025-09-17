# 🧠 模型訓練階段

這是機器學習專案的第一階段，負責訓練多種分類模型。

## 📋 功能特色

- **多種演算法支援**：KNN、隨機森林、SVM、PyTorch 神經網路
- **數據預處理**：自動標準化和數據分割
- **模型評估**：完整的性能評估指標
- **模型保存**：支援多種格式保存訓練好的模型

## 🚀 使用方法

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 運行訓練
```bash
python train_models.py
```

### 3. 查看結果
訓練完成後，模型將保存在 `../models/` 目錄中。

## 📊 支援的模型

| 模型 | 類型 | 特點 |
|------|------|------|
| KNN | 傳統ML | 簡單易懂，適合小數據集 |
| 隨機森林 | 集成學習 | 抗過擬合，特徵重要性 |
| SVM | 傳統ML | 高維數據，非線性分類 |
| PyTorch NN | 深度學習 | 複雜模式，可擴展性 |

## 📈 輸出文件

- `../models/knn_model.pkl` - KNN 模型
- `../models/random_forest_model.pkl` - 隨機森林模型
- `../models/svm_model.pkl` - SVM 模型
- `../models/pytorch_model.pth` - PyTorch 模型權重
- `../models/scaler.pkl` - 數據標準化器
- `../models/metadata.pkl` - 模型元數據
