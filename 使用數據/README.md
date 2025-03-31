## 範本數據集

- [**完整資料實作.ipynb**](./README.ipynb)

### forge數據集
- two-class classification(2個種類的分類)

```python
import mglearn
X, y = mglearn.datasets.make_forge()
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