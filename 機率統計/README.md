# 機率統計 

涉及隨機數生成、機率分佈、統計運算等應用，常用的函式庫包括 random（內建）、numpy、scipy.stats 和 pandas。以下是一些基本範例：


## 1. 隨機數生成（Random Sampling）

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

## 2. 常見機率分佈

### 2.1 常態分佈（Normal Distribution）

**numpy建立理論常態分佈**

```python
import numpy as np

# 生成 5 個符合均值 0、標準差 1 的正態分佈數值
normal_data = np.random.normal(loc=0, scale=1, size=5)
print(normal_data)
```


➜ 高斯分佈（Gaussian distribution），也稱為常態分佈（normal distribution），是一種連續概率分佈，常用於描述自然界中許多現象的數據分佈。它的特徵是數據分佈呈現鐘形曲線（bell-shaped curve），具有以下關鍵特性：

1. **對稱性**：曲線以平均值（mean, μ）為中心，左右對稱。
2. **集中性**：大多數數據點分佈在平均值附近，隨著數據值偏離平均值，出現的概率逐漸減少。
3. **參數**：由兩個參數決定：
	- 平均值（μ）：表示數據的中心位置。
	- 標準差（σ）: ：表示數據的分散程度，標準差越大，曲線越扁平。?

> [!IMPORTANT]
> 高斯分佈之所以重要，是因為許多自然現象（如身高、考試成績、測量誤差等）的數據分佈近似符合高斯分佈，這使得它在統計學和機器學習中廣泛應用。

**➜機器學習常說「特徵符合高斯分佈」**  

在機器學習中，特別是 GaussianNB（高斯單純貝氏模型），假設每個特徵的數據分佈符合高斯分佈，意味著：

- 該特徵的數值是連續的（例如身高、溫度）。
- 這些數值的分佈形狀近似鐘形曲線，大多數值集中在平均值附近，極端值（很小或很大）的出現概率較低。
- GaussianNB 使用高斯分佈的概率密度函數來估計每個特徵在不同類別下的概率，從而進行分類。

如果特徵的數據分佈偏離高斯分佈（例如呈現偏態或多峰分佈），GaussianNB 的表現可能會受到影響，因為它的假設不再完全成立。

**➜實際的例子**  

假設您正在研究學生的身高數據，來預測他們是否適合參加籃球隊：

- **特徵**：身高（連續數據，單位：厘米）。  
- **數據分佈**：假設身高數據符合高斯分佈，平均值 μ = 170 cm，標準差 σ = 10 cm。  
- **分佈特性**：  
	- 大多數學生身高集中在 160 cm 到 180 cm 之間（約 68% 的數據在 μ ± σ 範圍內）。  
	- 極端值（例如 140 cm 或 200 cm）的學生很少出現。  
	- 鐘形曲線：如果畫出身高分佈的直方圖，它會呈現一個對稱的鐘形曲線，中心在 170 cm。  

- **視學化理解**

1. 鐘形曲線：如果畫出身高分佈的直方圖，它會呈現一個對稱的鐘形曲線，中心在 170 cm。
2. 說明曲線下的面積表示概率，例如
	- 約 68% 的數據落在 μ ± σ 範圍內。
	- 約 95% 的數據落在 μ ± 2σ 範圍內。
3. 實際例子（如身高、考試成績）對照，強調「大多數數據集中在中間，極端值少見」。

**➜實作範例**

[**正態(常態)分佈實作ipynb檔**](./正態分佈.ipynb)

[**使用Q-Q圖(Quantile-Quantile Plot)是否為常態分佈-ipynb檔**](./正態分佈1.ipynb)



### 2.2. 均勻分佈（Uniform Distribution）

[**均勻分佈實作ipynb檔**](./均勻分佈.ipynb)

```other
uniform_data = np.random.uniform(low=0, high=10, size=5)
print(uniform_data)
```


### 2.3.  二項分佈（Binomial Distribution）

[**2項分佈實作ipynb檔**](./2項分佈.ipynb)

```other
# 進行 10 次獨立拋硬幣試驗，成功機率為 0.5
binomial_data = np.random.binomial(n=10, p=0.5, size=5)
print(binomial_data)
```

----

## 3. [Quantile-Quantile Plot](./Q-Q圖)（簡稱 Q-Q 圖）

是一種圖形工具，用於檢查數據的分佈是否符合某個特定的理論分佈，或者比較兩個數據集的分佈是否相似。

---



## 4. 基本統計計算

**使用 numpy 計算常見統計量**：

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


