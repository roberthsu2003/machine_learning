import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def train_and_save_model(
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 1.0,
    model_path: str = "diabetes_model.joblib"
) -> dict:
    """
    訓練糖尿病邏輯迴歸分類器並序列化模型與預處理器。
    """
    data_file = "Diabetes_Data.csv"
    if not os.path.exists(data_file):
        data_file = os.path.join(os.path.dirname(__file__), "Diabetes_Data.csv")
        
    print(f"正在載入糖尿病數據集：{data_file}...")
    df = pd.read_csv(data_file)
    
    # 類別轉換：性別 (男生: 1, 女生: 0)
    df_processed = df.copy()
    df_processed["Gender"] = (df_processed["Gender"] == "男生").astype(int)
    
    feature_names = ["Age", "Weight", "BloodSugar", "Gender"]
    feature_names_zh = ["年齡 (Age)", "體重 (Weight)", "空腹血糖 (BloodSugar)", "性別 (Gender)"]
    
    X = df_processed[feature_names]
    y = df_processed["Diabetes"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"開始訓練邏輯迴歸模型 (C={C})...")
    start_time = time.time()
    
    model = LogisticRegression(C=C, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"訓練完成！訓練集準確率: {train_acc:.4f}，測試集準確率: {test_acc:.4f}，耗時: {train_time:.4f}秒")
    
    # 計算勝算比 (Odds Ratio)
    coef = model.coef_[0]
    odds_ratios = np.exp(coef)
    or_dict = {
        name: float(val) for name, val in zip(feature_names_zh, odds_ratios)
    }
    
    metadata = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_names_zh": feature_names_zh,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "odds_ratios": or_dict,
        "train_time": float(train_time),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(metadata, model_path)
    print(f"模型與元數據已成功序列化至 {model_path}！")
    return metadata


if __name__ == "__main__":
    train_and_save_model()
