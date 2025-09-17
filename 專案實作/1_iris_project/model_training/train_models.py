"""
主訓練腳本
訓練多種機器學習模型並保存結果
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 導入自定義模組
from models import IrisClassifier, create_model, count_parameters
from data_loader import IrisDataLoader, evaluate_model

class ModelTrainer:
    """
    模型訓練器類
    負責訓練多種機器學習模型
    """
    
    def __init__(self, data_loader):
        """
        初始化訓練器
        
        Args:
            data_loader: 數據載入器實例
        """
        self.data_loader = data_loader
        self.models = {}
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用設備: {self.device}")
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test):
        """
        訓練傳統機器學習模型
        
        Args:
            X_train: 訓練特徵
            X_test: 測試特徵
            y_train: 訓練標籤
            y_test: 測試標籤
        """
        print("\n🤖 開始訓練傳統機器學習模型...")
        
        # 定義模型配置
        model_configs = {
            'knn': {
                'model': KNeighborsClassifier(n_neighbors=3),
                'name': 'K-近鄰分類器'
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'name': '隨機森林分類器'
            },
            'svm': {
                'model': SVC(probability=True, random_state=42),
                'name': '支援向量機'
            }
        }
        
        # 訓練每個模型
        for model_key, config in model_configs.items():
            print(f"\n📚 訓練 {config['name']}...")
            
            # 訓練模型
            model = config['model']
            model.fit(X_train, y_train)
            
            # 預測
            y_pred = model.predict(X_test)
            
            # 評估
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ {config['name']} 準確率: {accuracy:.4f}")
            
            # 保存模型和結果
            self.models[model_key] = model
            self.results[model_key] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model_name': config['name']
            }
    
    def train_pytorch_model(self, X_train, X_test, y_train, y_test, 
                           epochs=1000, learning_rate=0.01, model_type='complex'):
        """
        訓練 PyTorch 神經網路模型
        
        Args:
            X_train: 訓練特徵
            X_test: 測試特徵
            y_train: 訓練標籤
            y_test: 測試標籤
            epochs: 訓練輪數
            learning_rate: 學習率
            model_type: 模型類型
        """
        print(f"\n🧠 開始訓練 PyTorch 神經網路模型 ({model_type})...")
        
        # 轉換為 PyTorch 張量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # 創建模型
        model = create_model(model_type=model_type).to(self.device)
        print(f"📊 模型參數數量: {count_parameters(model)}")
        
        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 訓練模型
        model.train()
        train_losses = []
        
        print(f"🔄 開始訓練 ({epochs} 輪)...")
        for epoch in tqdm(range(epochs), desc="訓練進度"):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # 每100輪顯示一次損失
            if (epoch + 1) % 100 == 0:
                print(f"輪次 {epoch+1}/{epochs}, 損失: {loss.item():.4f}")
        
        # 評估模型
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ PyTorch 模型準確率: {accuracy:.4f}")
        
        # 保存模型和結果
        self.models['pytorch'] = model
        self.results['pytorch'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'model_name': 'PyTorch 神經網路',
            'train_losses': train_losses
        }
        
        return train_losses
    
    def save_models(self, save_path='../models/'):
        """
        保存所有訓練好的模型
        
        Args:
            save_path: 保存路徑
        """
        print(f"\n💾 保存模型到 {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        
        # 保存傳統模型
        for name, model in self.models.items():
            if name != 'pytorch':
                model_path = os.path.join(save_path, f'{name}_model.pkl')
                joblib.dump(model, model_path)
                print(f"✅ {name.upper()} 模型已保存")
        
        # 保存 PyTorch 模型
        if 'pytorch' in self.models:
            pytorch_path = os.path.join(save_path, 'pytorch_model.pth')
            torch.save(self.models['pytorch'].state_dict(), pytorch_path)
            print(f"✅ PyTorch 模型已保存")
        
        # 保存標準化器
        self.data_loader.save_scaler(save_path)
        
        print(f"✅ 所有模型已保存到 {save_path}")
    
    def create_performance_report(self, target_names):
        """
        創建性能報告
        
        Args:
            target_names: 類別名稱
        """
        print("\n📊 模型性能比較報告")
        print("=" * 50)
        
        # 按準確率排序
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (model_key, result) in enumerate(sorted_results, 1):
            print(f"{i}. {result['model_name']}: {result['accuracy']:.4f}")
        
        # 找出最佳模型
        best_model = sorted_results[0]
        print(f"\n🏆 最佳模型: {best_model[1]['model_name']} (準確率: {best_model[1]['accuracy']:.4f})")
        
        return sorted_results
    
    def plot_training_curves(self, save_path='../models/'):
        """
        繪製訓練曲線
        
        Args:
            save_path: 保存路徑
        """
        if 'pytorch' in self.results and 'train_losses' in self.results['pytorch']:
            # 確保目錄存在
            os.makedirs(save_path, exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.results['pytorch']['train_losses'])
            plt.title('PyTorch Model Training Loss Curve', fontsize=14)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            
            curve_path = os.path.join(save_path, 'training_curve.png')
            plt.savefig(curve_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 訓練曲線已保存到 {curve_path}")

def main():
    """
    主訓練流程
    """
    print("🚀 開始機器學習模型訓練流程...")
    print("=" * 60)
    
    # 初始化數據載入器
    data_loader = IrisDataLoader()
    
    # 載入數據
    print("📥 載入 Iris 數據集...")
    iris_data = data_loader.load_data()
    
    # 顯示數據摘要
    summary = data_loader.get_data_summary(iris_data)
    print(f"✅ 數據載入完成:")
    print(f"   - 樣本數: {summary['n_samples']}")
    print(f"   - 特徵數: {summary['n_features']}")
    print(f"   - 類別數: {summary['n_classes']}")
    print(f"   - 類別分佈: {summary['class_distribution']}")
    
    # 分割數據
    print("\n✂️ 分割數據...")
    X_train, X_test, y_train, y_test = data_loader.split_data(
        iris_data['data'], iris_data['target']
    )
    print(f"✅ 數據分割完成:")
    print(f"   - 訓練集: {X_train.shape}")
    print(f"   - 測試集: {X_test.shape}")
    
    # 標準化數據
    print("\n🔧 標準化數據...")
    X_train_scaled = data_loader.fit_transform_data(X_train)
    X_test_scaled = data_loader.transform_data(X_test)
    print("✅ 數據標準化完成")
    
    # 初始化訓練器
    trainer = ModelTrainer(data_loader)
    
    # 訓練傳統模型
    trainer.train_traditional_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 訓練 PyTorch 模型
    trainer.train_pytorch_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 創建性能報告
    trainer.create_performance_report(iris_data['target_names'])
    
    # 繪製訓練曲線
    trainer.plot_training_curves()
    
    # 保存所有模型
    trainer.save_models()
    
    # 保存元數據
    data_loader.save_metadata(iris_data)
    
    print("\n🎉 模型訓練完成！")
    print("=" * 60)
    print("📁 生成的文件:")
    print("   - ../models/knn_model.pkl")
    print("   - ../models/random_forest_model.pkl") 
    print("   - ../models/svm_model.pkl")
    print("   - ../models/pytorch_model.pth")
    print("   - ../models/scaler.pkl")
    print("   - ../models/metadata.pkl")
    print("   - ../models/training_curve.png")

if __name__ == "__main__":
    main()
