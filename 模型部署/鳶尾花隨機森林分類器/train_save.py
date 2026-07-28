import os
import time
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save_model(
    n_estimators: int = 100,
    max_depth: int = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    訓練隨機森林分類器並序列化模型。
    
    參數:
        n_estimators: 樹的數量
        max_depth: 樹的最大深度
        test_size: 測試集比例 (0.1 ~ 0.5)
        random_state: 隨機種子
        
    回傳:
        包含訓練指標、特徵重要性與花費時間的字典。
    """
    print(f"正在載入 Iris 數據集...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = [name.replace(" (cm)", "") for name in iris.feature_names]

    # 切分訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"開始訓練隨機森林分類器 (樹數量: {n_estimators}, 最大深度: {max_depth})...")
    start_time = time.time()
    
    # 建立並訓練模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth is not None and max_depth > 0 else None,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time

    # 計算測試集準確度
    accuracy = model.score(X_test, y_test)
    print(f"模型訓練完成！測試集準確度 (Accuracy): {accuracy:.4f}，耗時: {train_time:.4f}秒")

    # 取得特徵重要性
    importances = model.feature_importances_
    feature_importances = {
        name: float(imp) for name, imp in zip(feature_names, importances)
    }

    # 儲存模型以及所有相關元數據 (Metadata)
    model_data = {
        "model": model,
        "target_names": list(iris.target_names),
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "accuracy": float(accuracy),
        "train_time": float(train_time),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "random_state": random_state
    }

    # 取得當前腳本所在的目錄，並組合出模型路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(current_dir, "iris_model.joblib")
    
    print(f"正在將模型與元數據序列化並儲存至 {model_filename}...")
    joblib.dump(model_data, model_filename)
    print("模型儲存成功！")
    
    return {
        "status": "success",
        "accuracy": float(accuracy),
        "train_time": float(train_time),
        "feature_importances": feature_importances,
        "message": "模型訓練完成並儲存成功！"
    }


if __name__ == "__main__":
    train_and_save_model()

