# 多元線性迴歸(Multiple Linear degression)

> [!TIP]
> [Jam簡報檔目錄](./說明jam)  
> [ipynb實作](./multiple_linear_regression1.ipynb)

## 讀取資料

```python
import pandas as pd

url = "Salary_Data2.csv"
data = pd.read_csv(url)
data
```

![](./images/pic1.png)

## Label Encoding
- 將文字轉為數值
- 有高低的欄位(EducationLevel)
- 高中以下 -> 0
- 大學 -> 1
- 碩士以上 -> 2


```python
educationLabel_encoding = data['EducationLevel'].map({'高中以下':0,'大學':1,'碩士以上':2})

data['EducationLevel'] = educationLabel_encoding
```

![](./images/pic2.png)

## One Hot Encoding
- 將文字轉為數值
- 沒有高低的欄位(City)
- 將城市A,城市B,城市C的值轉換為欄位
- 刪除City,CityC

```pyhton
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(categories=[['城市A', '城市B', '城市C']], handle_unknown='ignore') #如果出現新的城市，handle_unknown='ignore' 至關重要，以避免錯誤

#轉換過程必需是2維的資料
#由serirec轉換為DataFrame
city_encoded = onehot_encoder.fit_transform(data[['City']],)

# city_encoded是一個稀疏矩陣,轉換為nd_array
# 增加資料至data內
data[['CityA','CityB','CityC']] = city_encoded.toarray()

#刪除欄位
data = data.drop(['City','CityC'],axis=1)

```

![](./images/pic3.png)


## 手動實作
- cost function
- gradient descent
