import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def train_and_save_model(
    test_size: float = 0.2,
    random_state: int = 76
) -> dict:
    """
    訓練多元線性迴歸模型以預測薪資，並將模型與預處理器序列化儲存。
    
    參數:
        test_size: 測試集比例 (0.1 ~ 0.5)
        random_state: 隨機種子 (預設 76，與教學 Notebook 一致)
        
    回傳:
        包含訓練指標、權重與花費時間的字典。
    """
    print("正在載入薪資數據集...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Salary_Data2.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到數據集檔案: {csv_path}")
        
    data = pd.read_csv(csv_path)
    
    start_time = time.time()
    
    # 1. 建立並擬合 LabelEncoder (學歷：大學=0, 碩士以上=1, 高中以下=2)
    # 我們顯式指定類別，確保不論訓練集如何切分，編碼對照永遠一致：
    # le.classes_ 順序會是 ['大學', '碩士以上', '高中以下']
    # 這是因為 fit 會對傳入清單做不重覆排序，中文字排序為：大學 (idx 0), 碩士以上 (idx 1), 高中以下 (idx 2)
    le = LabelEncoder()
    le.fit(["大學", "碩士以上", "高中以下"])
    data['EducationLevel'] = le.transform(data['EducationLevel'])
    
    # 2. 建立並擬合 OneHotEncoder (城市：城市A, 城市B, 城市C)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # 建立一個範例 DataFrame 確保擬合順序固定
    ohe.fit(pd.DataFrame([["城市A"], ["城市B"], ["城市C"]], columns=["City"]))
    
    city_encoded = ohe.transform(data[['City']])
    city_cols = ohe.get_feature_names_out(['City'])
    city_df = pd.DataFrame(city_encoded, columns=city_cols)
    
    # 拼接特徵並刪除原始的 City 欄位
    data = pd.concat([data, city_df], axis=1).drop('City', axis=1)
    
    # 定義特徵欄位與目標變數
    feature_names = ['YearsExperience', 'EducationLevel', 'City_城市A', 'City_城市B', 'City_城市C']
    X = data[feature_names]
    y = data['Salary']
    
    # 3. 切分訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 4. 特徵標準化 (對所有特徵進行)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"開始訓練多元線性迴歸模型 (測試集比例: {test_size}, 隨機種子: {random_state})...")
    
    # 5. 建立並訓練線性迴歸模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    
    # 計算測試集 R-squared (決定係數)
    r2 = model.score(X_test_scaled, y_test)
    print(f"模型訓練完成！測試集 R-squared: {r2:.4f}，耗時: {train_time:.4f}秒")
    
    # 取得權重 (Coefficients) 與截距 (Intercept)
    coefs = model.coef_
    intercept = model.intercept_
    
    feature_coefs = {
        name: float(coef) for name, coef in zip(feature_names, coefs)
    }
    
    # 6. 儲存模型以及所有相關預處理器與元數據 (Metadata)
    model_data = {
        "model": model,
        "le": le,
        "ohe": ohe,
        "scaler": scaler,
        "r2": float(r2),
        "coef": [float(c) for c in coefs],
        "intercept": float(intercept),
        "feature_names": feature_names,
        "feature_coefs": feature_coefs,
        "train_time": float(train_time),
        "test_size": test_size,
        "random_state": random_state
    }
    
    model_filename = os.path.join(current_dir, "salary_model.joblib")
    print(f"正在將模型、預處理器與元數據序列化並儲存至 {model_filename}...")
    joblib.dump(model_data, model_filename)
    print("模型儲存成功！")
    
    return {
        "status": "success",
        "r2": float(r2),
        "coef": [float(c) for c in coefs],
        "intercept": float(intercept),
        "feature_coefs": feature_coefs,
        "train_time": float(train_time),
        "message": "模型訓練完成並儲存成功！"
    }


if __name__ == "__main__":
    train_and_save_model()
