## numpy在機器學習中常用的操作
> [!IMPORTANT]
> - 包含numpy在機器學習中常用的操作
> - 包含簡單的數據集
> - 包含簡單圖表的顯示  
> - [**numpy的基本操作和ipynb**](./README.ipynb)

---

## 範本數據集
### forge數據集

- two-class classification(2個種類的分類)

> [!TIP]
> [**forge數據集實作ipynb**](./forge數據集.ipynb)

```python
import mglearn
X, y = mglearn.datasets.make_forge()

```

---
### wave數據集
- regression 迴歸的演算法
- X:1個feature
- y:1個label

> [!TIP]
> [**wave數據集實作ipynb**](./wave數據集.ipynb)

```python
import mglearn
X, y = mglearn.datasets.make_wave(n_samples=40)
```

---

### Scikit-learn的Iris 數據
- Scikit-learn 內建的 Iris (鳶尾花) 數據集

> [!TIP]
> [**Iris數據集實作ipynb**](./Iris數據集進行數據選擇與切片.ipynb)

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

---
### Wisconsin Breast Cancer dataset(威斯康辛州乳癌資料集)

> [!TIP]
> [**威斯康辛州乳癌資料集數據集實作ipynb**](./威斯康辛州乳癌數據集_load_breast_cancer.ipynb)

```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
```

---
### 加州房價數據集 (Scikit-learn) 與 Ames 房價數據集 (OpenML)

> [!TIP]
> [**加州房價數據集和Ames 房價數據集 ipynb**](./加州房價數據集_fetch_california_housing.ipynb)


```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(f"California housing data shape: {housing.data.shape}")


from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
print(f"Ames housing data shape: {housing.data.shape}")
```

---
### UCI 機器學習庫
UCI：加州大學歐文分校的縮寫
UCI 機器學習庫是一個非常著名且廣泛使用的資料庫、領域理論和資料產生器的集合，機器學習社群使用它來進行機器學習演算法的實證分析。

https://archive.ics.uci.edu/datasets

---

### Kaggle Datasets
Kaggle 是一個非常知名的平台，尤其在數據科學和機器學習領域。可以把它想像成一個集多功能於一身的社群和資源中心。

https://www.kaggle.com/datasets

