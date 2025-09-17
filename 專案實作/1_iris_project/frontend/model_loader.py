"""
模型載入器
負責載入訓練好的模型並提供預測接口
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from typing import Dict, List, Tuple, Optional, Any

# 添加模型訓練目錄到路徑
sys.path.append('../model_training')
from models import IrisClassifier, create_model

class ModelLoader:
    """
    模型載入器類
    負責載入和管理所有訓練好的模型
    """
    
    def __init__(self, models_path='../models/'):
        """
        初始化模型載入器
        
        Args:
            models_path: 模型文件路徑
        """
        self.models_path = models_path
        self.models = {}
        self.scaler = None
        self.metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型配置
        self.model_configs = {
            'knn': {'name': 'K-近鄰分類器', 'type': 'traditional'},
            'random_forest': {'name': '隨機森林', 'type': 'traditional'},
            'svm': {'name': '支援向量機', 'type': 'traditional'},
            'pytorch': {'name': 'PyTorch 神經網路', 'type': 'pytorch'}
        }
    
    def load_all_models(self) -> bool:
        """
        載入所有訓練好的模型
        
        Returns:
            bool: 是否載入成功
        """
        try:
            print("🔄 載入模型...")
            
            # 載入元數據
            self.metadata = joblib.load(os.path.join(self.models_path, 'metadata.pkl'))
            print("✅ 元數據載入成功")
            
            # 載入標準化器
            self.scaler = joblib.load(os.path.join(self.models_path, 'scaler.pkl'))
            print("✅ 標準化器載入成功")
            
            # 載入傳統模型
            traditional_models = ['knn', 'random_forest', 'svm']
            for model_name in traditional_models:
                model_path = os.path.join(self.models_path, f'{model_name}_model.pkl')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"✅ {self.model_configs[model_name]['name']} 載入成功")
                else:
                    print(f"⚠️ 找不到 {model_name} 模型文件")
            
            # 載入 PyTorch 模型
            pytorch_path = os.path.join(self.models_path, 'pytorch_model.pth')
            if os.path.exists(pytorch_path):
                pytorch_model = create_model(model_type='complex')
                pytorch_model.load_state_dict(torch.load(pytorch_path, map_location=self.device))
                pytorch_model.eval()
                pytorch_model.to(self.device)
                self.models['pytorch'] = pytorch_model
                print(f"✅ {self.model_configs['pytorch']['name']} 載入成功")
            else:
                print("⚠️ 找不到 PyTorch 模型文件")
            
            print(f"🖥️ 使用設備: {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ 模型載入失敗: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        獲取可用的模型列表
        
        Returns:
            List[str]: 可用模型名稱列表
        """
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        獲取模型資訊
        
        Returns:
            Dict: 模型資訊字典
        """
        info = {}
        for model_name in self.get_available_models():
            info[model_name] = {
                'name': self.model_configs[model_name]['name'],
                'type': self.model_configs[model_name]['type'],
                'available': True
            }
        return info
    
    def predict_single_model(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        使用單個模型進行預測
        
        Args:
            model_name: 模型名稱
            X: 輸入特徵 (已標準化)
            
        Returns:
            Tuple: (預測類別, 預測機率)
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        
        if model_name == 'pytorch':
            # PyTorch 模型預測
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = torch.max(outputs, 1)
                predictions = predicted.cpu().numpy()
        else:
            # 傳統模型預測
            predictions = model.predict(X)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
            else:
                probabilities = None
        
        return predictions, probabilities
    
    def predict_all_models(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        使用所有可用模型進行預測
        
        Args:
            X: 輸入特徵 (未標準化)
            
        Returns:
            Dict: 所有模型的預測結果
        """
        # 標準化輸入
        X_scaled = self.scaler.transform(X)
        
        results = {}
        for model_name in self.get_available_models():
            try:
                predictions, probabilities = self.predict_single_model(model_name, X_scaled)
                
                results[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'model_name': self.model_configs[model_name]['name'],
                    'type': self.model_configs[model_name]['type']
                }
            except Exception as e:
                print(f"⚠️ 模型 {model_name} 預測失敗: {str(e)}")
                results[model_name] = {
                    'predictions': None,
                    'probabilities': None,
                    'model_name': self.model_configs[model_name]['name'],
                    'type': self.model_configs[model_name]['type'],
                    'error': str(e)
                }
        
        return results
    
    def get_target_names(self) -> List[str]:
        """
        獲取類別名稱
        
        Returns:
            List[str]: 類別名稱列表
        """
        if self.metadata is None:
            return ['山鳶尾', '變色鳶尾', '維吉尼亞鳶尾']
        return self.metadata['target_names']
    
    def get_feature_names(self) -> List[str]:
        """
        獲取特徵名稱
        
        Returns:
            List[str]: 特徵名稱列表
        """
        if self.metadata is None:
            return ['萼片長度 (公分)', '萼片寬度 (公分)', '花瓣長度 (公分)', '花瓣寬度 (公分)']
        return self.metadata['feature_names']
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        獲取數據摘要
        
        Returns:
            Dict: 數據摘要資訊
        """
        if self.metadata is None:
            return {
                'n_samples': 150,
                'n_features': 4,
                'n_classes': 3,
                'feature_names': self.get_feature_names(),
                'target_names': self.get_target_names()
            }
        return self.metadata['data_summary']
    
    def analyze_predictions(self, predictions_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析預測結果
        
        Args:
            predictions_dict: 預測結果字典
            
        Returns:
            Dict: 分析結果
        """
        # 提取所有預測結果
        pred_values = {}
        for model_name, result in predictions_dict.items():
            if result['predictions'] is not None:
                pred_values[model_name] = result['predictions'][0]
        
        if not pred_values:
            return {'consensus': False, 'unique_predictions': [], 'agreement_rate': 0.0}
        
        # 計算一致性
        unique_predictions = list(set(pred_values.values()))
        consensus = len(unique_predictions) == 1
        
        # 計算同意率
        if len(pred_values) > 1:
            most_common = max(set(pred_values.values()), key=list(pred_values.values()).count)
            agreement_rate = list(pred_values.values()).count(most_common) / len(pred_values)
        else:
            agreement_rate = 1.0
        
        # 按預測分組
        prediction_groups = {}
        for model_name, pred in pred_values.items():
            pred_name = self.get_target_names()[pred]
            if pred_name not in prediction_groups:
                prediction_groups[pred_name] = []
            prediction_groups[pred_name].append(model_name)
        
        return {
            'consensus': consensus,
            'unique_predictions': unique_predictions,
            'agreement_rate': agreement_rate,
            'prediction_groups': prediction_groups,
            'predictions': pred_values
        }

# 全局模型載入器實例
model_loader = ModelLoader()

def load_models() -> bool:
    """
    載入所有模型（全局函數）
    
    Returns:
        bool: 是否載入成功
    """
    return model_loader.load_all_models()

def get_model_loader() -> ModelLoader:
    """
    獲取模型載入器實例
    
    Returns:
        ModelLoader: 模型載入器實例
    """
    return model_loader

if __name__ == "__main__":
    # 測試模型載入器
    loader = ModelLoader()
    
    if loader.load_all_models():
        print("\n✅ 所有模型載入成功!")
        
        # 測試預測
        test_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # 山鳶尾範例
        results = loader.predict_all_models(test_data)
        
        print("\n🔮 測試預測結果:")
        for model_name, result in results.items():
            if result['predictions'] is not None:
                pred_class = loader.get_target_names()[result['predictions'][0]]
                print(f"{result['model_name']}: {pred_class}")
            else:
                print(f"{result['model_name']}: 預測失敗")
        
        # 分析預測一致性
        analysis = loader.analyze_predictions(results)
        print(f"\n📊 預測一致性: {'是' if analysis['consensus'] else '否'}")
        print(f"同意率: {analysis['agreement_rate']:.2%}")
    else:
        print("❌ 模型載入失敗!")
