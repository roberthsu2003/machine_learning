import pandas as pd

# 建立一個包含不一致和重複數據的範例 DataFrame
data = {'age': ['25', '30', '25', '35', '30', '40', '40'],
        'gender': ['Male', 'Female', 'male', 'Female', 'Male', 'Female', 'Female'],
        'salary': [50000, 60000, 50000, 70000, 60000, 80000, 80000]};
df = pd.DataFrame(data)

print("原始數據:")
print(df)
print("\n")

# --- 數據清洗步驟 ---

# 1. 處理重複數據
# duplicated() 會標記重複的行
print("檢查重複數據:")
print(df.duplicated())
print("\n")

# drop_duplicates() 會刪除重複的行
df_no_duplicates = df.drop_duplicates()
print("刪除重複數據後:")
print(df_no_duplicates)
print("\n")


# 2. 處理數據格式不一致
# 將 'age' 欄位轉換為數值型態
df_cleaned = df_no_duplicates.copy()
df_cleaned['age'] = pd.to_numeric(df_cleaned['age'])
print("將 'age' 轉換為數值型態:")
print(df_cleaned.dtypes)
print("\n")

# 3. 處理類別數據不一致
# 將 'gender' 欄位統一轉換為小寫
df_cleaned['gender'] = df_cleaned['gender'].str.lower()
print("將 'gender' 統一為小寫:")
print(df_cleaned)
print("\n")

# 現在 'gender' 欄位的值變為一致
print("清洗後的 'gender' 欄位唯一值:")
print(df_cleaned['gender'].unique())
