"""
數據載入和預處理模組
負責載入 Iris 數據集並進行預處理
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class IrisDataLoader:
    """
    Iris 數據集載入和預處理類
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        初始化數據載入器
        
        Args:
            test_size: 測試集比例
            random_state: 隨機種子
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 中文名稱映射
        self.chinese_names = {
            'setosa': '山鳶尾',
            'versicolor': '變色鳶尾',
            'virginica': '維吉尼亞鳶尾'
        }
        
        self.chinese_feature_names = {
            'sepal length (cm)': '萼片長度 (公分)',
            'sepal width (cm)': '萼片寬度 (公分)',
            'petal length (cm)': '花瓣長度 (公分)',
            'petal width (cm)': '花瓣寬度 (公分)'
        }
    
    def load_data(self):
        """
        載入 Iris 數據集
        
        Returns:
            dict: 包含數據和元數據的字典
        """
        # 載入原始數據
        iris = load_iris()
        
        # 創建中文版本
        iris_data = {
            'data': iris.data,
            'target': iris.target,
            'target_names': [self.chinese_names[name] for name in iris.target_names],
            'feature_names': [self.chinese_feature_names[name] for name in iris.feature_names],
            'DESCR': iris.DESCR,
            'original_target_names': iris.target_names,
            'original_feature_names': iris.feature_names
        }
        
        return iris_data
    
    def split_data(self, X, y, stratify=True):
        """
        分割數據為訓練集和測試集
        
        Args:
            X: 特徵數據
            y: 標籤數據
            stratify: 是否使用分層抽樣
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train):
        """
        擬合標準化器
        
        Args:
            X_train: 訓練數據
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
    
    def transform_data(self, X):
        """
        標準化數據
        
        Args:
            X: 要標準化的數據
            
        Returns:
            標準化後的數據
        """
        if not self.is_fitted:
            raise ValueError("標準化器尚未擬合，請先調用 fit_scaler")
        
        return self.scaler.transform(X)
    
    def fit_transform_data(self, X_train):
        """
        擬合並轉換訓練數據
        
        Args:
            X_train: 訓練數據
            
        Returns:
            標準化後的訓練數據
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        return X_train_scaled
    
    def get_data_summary(self, iris_data):
        """
        獲取數據摘要資訊
        
        Args:
            iris_data: 數據字典
            
        Returns:
            dict: 數據摘要
        """
        summary = {
            'n_samples': iris_data['data'].shape[0],
            'n_features': iris_data['data'].shape[1],
            'n_classes': len(iris_data['target_names']),
            'class_distribution': np.bincount(iris_data['target']),
            'feature_names': iris_data['feature_names'],
            'target_names': iris_data['target_names']
        }
        
        return summary
    
    def create_dataframe(self, X, y, feature_names, target_names):
        """
        創建 DataFrame 用於視覺化
        
        Args:
            X: 特徵數據
            y: 標籤數據
            feature_names: 特徵名稱
            target_names: 類別名稱
            
        Returns:
            pd.DataFrame: 包含特徵和標籤的 DataFrame
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['品種'] = [target_names[i] for i in y]
        return df
    
    def save_metadata(self, iris_data, save_path='../models/'):
        """
        保存數據元數據
        
        Args:
            iris_data: 數據字典
            save_path: 保存路徑
        """
        os.makedirs(save_path, exist_ok=True)
        
        metadata = {
            'target_names': iris_data['target_names'],
            'feature_names': iris_data['feature_names'],
            'original_target_names': iris_data['original_target_names'],
            'original_feature_names': iris_data['original_feature_names'],
            'data_summary': self.get_data_summary(iris_data)
        }
        
        joblib.dump(metadata, os.path.join(save_path, 'metadata.pkl'))
        print(f"✅ 元數據已保存到 {save_path}metadata.pkl")
    
    def save_scaler(self, save_path='../models/'):
        """
        保存標準化器
        
        Args:
            save_path: 保存路徑
        """
        if not self.is_fitted:
            raise ValueError("標準化器尚未擬合")
        
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
        print(f"✅ 標準化器已保存到 {save_path}scaler.pkl")

def evaluate_model(y_true, y_pred, target_names, model_name="模型"):
    """
    評估模型性能
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        target_names: 類別名稱
        model_name: 模型名稱
        
    Returns:
        dict: 評估結果
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n📊 {model_name} 評估結果:")
    print(f"準確率: {accuracy:.4f}")
    
    # 分類報告
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # 混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }
    
    return results

if __name__ == "__main__":
    # 測試數據載入器
    loader = IrisDataLoader()
    iris_data = loader.load_data()
    
    print("🌸 Iris 數據集載入成功!")
    print(f"樣本數: {iris_data['data'].shape[0]}")
    print(f"特徵數: {iris_data['data'].shape[1]}")
    print(f"類別數: {len(iris_data['target_names'])}")
    print(f"類別名稱: {iris_data['target_names']}")
    print(f"特徵名稱: {iris_data['feature_names']}")
    
    # 測試數據分割
    X_train, X_test, y_train, y_test = loader.split_data(iris_data['data'], iris_data['target'])
    print(f"\n數據分割完成:")
    print(f"訓練集: {X_train.shape}")
    print(f"測試集: {X_test.shape}")
    
    # 測試標準化
    X_train_scaled = loader.fit_transform_data(X_train)
    X_test_scaled = loader.transform_data(X_test)
    print(f"\n標準化完成:")
    print(f"訓練集標準化後形狀: {X_train_scaled.shape}")
    print(f"測試集標準化後形狀: {X_test_scaled.shape}")
