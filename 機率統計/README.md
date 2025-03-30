## 機率統計 

涉及隨機數生成、機率分佈、統計運算等應用，常用的函式庫包括 random（內建）、numpy、scipy.stats 和 pandas。以下是一些基本範例：


**1. 隨機數生成（Random Sampling）**

使用 random 來產生隨機數：

```other
import random

# 生成 0 到 1 之間的隨機浮點數
print(random.random())

# 生成 1 到 100 之間的隨機整數
print(random.randint(1, 100))

# 從列表中隨機選擇一個元素
choices = ['蘋果', '香蕉', '橘子']
print(random.choice(choices))

# 從列表中抽取 2 個不重複的元素
print(random.sample(choices, 2))
```

----

**2. 常見機率分佈**

使用 numpy 產生符合不同機率分佈的隨機數。

**(1) 正態分佈（Normal Distribution）**

### 正態分佈實作檔

[**正態(常態)分佈實作ipynb檔**](./正態分佈.ipynb)

[**正態(常態)分佈實作ipynb檔**](./正態分佈1.ipynb)



```other
import numpy as np

# 生成 5 個符合均值 0、標準差 1 的正態分佈數值
normal_data = np.random.normal(loc=0, scale=1, size=5)
print(normal_data)
```


**(2) 均勻分佈（Uniform Distribution）**

[**均勻分佈實作ipynb檔**](./均勻分佈.ipynb)

```other
uniform_data = np.random.uniform(low=0, high=10, size=5)
print(uniform_data)
```


**(3) 二項分佈（Binomial Distribution）**

[**2項分佈實作ipynb檔**](./2項分佈.ipynb)

```other
# 進行 10 次獨立拋硬幣試驗，成功機率為 0.5
binomial_data = np.random.binomial(n=10, p=0.5, size=5)
print(binomial_data)
```

----

**3. 基本統計計算**

使用 numpy 計算常見統計量：

```other
data = [12, 15, 20, 25, 30]

# 平均數
print("平均數:", np.mean(data))

# 中位數
print("中位數:", np.median(data))

# 標準差
print("標準差:", np.std(data))

# 變異數
print("變異數:", np.var(data))
```

----

**4. 機率密度函數與累積分佈函數**

使用 scipy.stats 計算機率分佈相關函數。

**(1) 計算某數值在標準正態分佈下的機率**

```other
from scipy.stats import norm

x = 1.0  # 目標數值
pdf_value = norm.pdf(x, loc=0, scale=1)  # 機率密度函數
cdf_value = norm.cdf(x, loc=0, scale=1)  # 累積分佈函數

print("PDF:", pdf_value)
print("CDF:", cdf_value)
```


**(2) 計算某數值在二項分佈下的機率**

```other
from scipy.stats import binom

n, p, k = 10, 0.5, 3  # 10 次試驗, 成功機率 0.5, 觀察 3 次成功
prob = binom.pmf(k, n, p)
print(f"成功 {k} 次的機率: {prob}")
```


