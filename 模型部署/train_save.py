import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save_model():
    print("正在載入 Iris 數據集...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # 切分訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("開始訓練隨機森林分類器 (Random Forest Classifier)...")
    # 建立並訓練模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 計算測試集準確度
    accuracy = model.score(X_test, y_test)
    print(f"模型訓練完成！測試集準確度 (Accuracy): {accuracy:.4f}")

    # 儲存模型以及類別名稱
    model_data = {"model": model, "target_names": list(iris.target_names)}

    # 取得當前腳本所在的目錄，並組合出模型路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(current_dir, "iris_model.joblib")
    
    print(f"正在將模型序列化並儲存至 {model_filename}...")
    joblib.dump(model_data, model_filename)
    print("模型儲存成功！")


if __name__ == "__main__":
    train_and_save_model()
