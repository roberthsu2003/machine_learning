import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save_model(
    n_estimators: int = 100,
    max_depth: int = 4,
    test_size: float = 0.25,
    random_state: int = 42,
    model_path: str = "heart_failure_model.joblib"
) -> dict:
    """
    訓練心臟衰竭隨機森林分類器並序列化模型。
    """
    data_file = "heart_failure_clinical_records_dataset.csv"
    if not os.path.exists(data_file):
        data_file = os.path.join(os.path.dirname(__file__), "heart_failure_clinical_records_dataset.csv")
        
    print(f"正在載入心臟衰竭數據集：{data_file}...")
    df = pd.read_csv(data_file)
    
    feature_names = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking'
    ]
    
    feature_names_zh = [
        '年齡 (age)', '貧血 (anaemia)', '肌酸激酶 (CPK)', '糖尿病 (diabetes)',
        '射血分數 (EF)', '高血壓 (HBP)', '血小板 (platelets)',
        '血清肌酸酐 (creatinine)', '血清鈉 (sodium)', '性別 (sex)', '抽菸 (smoking)'
    ]
    
    X = df[feature_names]
    y = df['DEATH_EVENT']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"開始訓練隨機森林 (n_estimators={n_estimators}, max_depth={max_depth})...")
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth and max_depth > 0 else None,
        min_samples_split=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"訓練完成！訓練集準確率: {train_acc:.4f}，測試集準確率: {test_acc:.4f}，耗時: {train_time:.4f}秒")
    
    # 特徵重要性
    importances = model.feature_importances_
    imp_dict = {
        name: float(val) for name, val in zip(feature_names_zh, importances)
    }
    
    metadata = {
        "model": model,
        "feature_names": feature_names,
        "feature_names_zh": feature_names_zh,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "feature_importances": imp_dict,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "train_time": float(train_time),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(metadata, model_path)
    print(f"模型與元數據已成功序列化至 {model_path}！")
    return metadata


if __name__ == "__main__":
    train_and_save_model()
