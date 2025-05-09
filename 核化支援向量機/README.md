# 支援向量機 SVC（Support Vector Classifier）

## 重點:
- 解釋支援向量機的基本概念，包括最大邊界、支援向量和核函數。
- 解釋核函數的意義，以及如何使用不同的核函數來處理非線性分類問題。
- 解釋懲罰參數 C 的意義，以及它如何影響模型的複雜度和泛化能力。
- 修改核函數和 C 值，觀察模型的變化。
- 介紹其他評估指標，例如精確度、召回率和 F1 分數。
- 嘗試使用其他數據集，例如手寫數字數據集，來進行 SVM 分類分析。
- 解釋 SVM 的優缺點，例如在高維空間中表現良好、但對參數敏感。
- 討論 SVM 的應用場景，例如圖像分類和文本分類。
- 解釋什麼是支持向量，以及支持向量在模型中的重要性。

## 進階:
- 不同核函數的原理，例如多項式核函數和徑向基核函數（RBF）。
- 可以講解如何使用交叉驗證來選擇最佳的參數組合。
- 如何使用 SVM 進行迴歸分析（SVR）。

### 何時使用 Support Vector Classifier(Linear SVC)?

**➜適合情況**:   
1. 資料在原始特徵空間中大致線性可分。
2. 資料維度(特徵數)遠大於樣本數(例如文字分類,TF-IDF)
3. 想要模型可解釋性較高
4. 速度考量: 線性 SVM 通常比 kernel SVM 快得多，尤其是資料量大時。
5. 想做特徵選擇(可搭配L1 Regularization)

```
from sklearn.svm import SVC
clf = SVC(kernel='linear')
```

### 何時使用 Support Vector Classifier(Linear SVC)?

**➜適合情況**:

1. 資料在原始空間中不是線性可分,需要更靈活的decision boundary
2. 資料數量不是很大(kernel方法計算量大)
3. 想利用RBF, Polynomial 或 Sigmoid Kernel 來捕捉非線性關係
4. 資料有複雜的形狀或結構(例如螺旋,環狀)

```python
from sklearn.svm import SVC
clf = SVC(keranl='rbf')
```

## 實作:
[**SVC(Support Vector Classifier）model實作**](./sklearn實作1.ipynb)