## 範本數據集
> [!IMPORTANT]
> - 包含numpy在機器學習中常用的操作
> - 包含常用的數據集
> - 包含圖表的顯示  
> - [**完整資料實作.ipynb**](./README.ipynb)

### forge數據集
- two-class classification(2個種類的分類)

```python
import mglearn
X, y = mglearn.datasets.make_forge()

#X是特徵(feature)一般用大寫的
#y是標籤(label)

X.shape, y.shape
#======output===========
((26, 2), (26,))
X有26筆資料,每筆資料有2個特徵
y有26個標籤

```

### wave數據集
- regression 迴歸的演算法
- X:1個feature
- y:1個label

```python
import mglearn
X, y = mglearn.datasets.make_wave(n_samples=40)
```

### Wisconsin Breast Cancer dataset(威斯康辛州乳癌資料集)

```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
```

### fetch_california_housing數據集

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(f"California housing data shape: {housing.data.shape}")
```

### Ames housing dataset數據集

```python
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
print(f"Ames housing data shape: {housing.data.shape}")
```


### UCI 機器學習庫
UCI：加州大學歐文分校的縮寫
UCI 機器學習庫是一個非常著名且廣泛使用的資料庫、領域理論和資料產生器的集合，機器學習社群使用它來進行機器學習演算法的實證分析。

https://archive.ics.uci.edu/datasets


### Kaggle Datasets
Kaggle 是一個非常知名的平台，尤其在數據科學和機器學習領域。可以把它想像成一個集多功能於一身的社群和資源中心。

https://www.kaggle.com/datasets

