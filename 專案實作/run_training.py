#!/usr/bin/env python3
"""
一鍵運行模型訓練腳本
簡化模型訓練流程
"""

import os
import sys
import subprocess
import time

def print_banner():
    """打印歡迎橫幅"""
    print("=" * 60)
    print("🤖 機器學習模型訓練啟動器")
    print("=" * 60)
    print()

def check_requirements():
    """檢查依賴是否安裝"""
    print("🔍 檢查依賴...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'torch', 
        'matplotlib', 'seaborn', 'joblib', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安裝")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下套件: {', '.join(missing_packages)}")
        print("請運行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依賴已安裝")
    return True

def run_training():
    """運行模型訓練"""
    print("\n🚀 開始模型訓練...")
    print("-" * 40)
    
    # 切換到模型訓練目錄
    training_dir = os.path.join(os.path.dirname(__file__), 'model_training')
    
    if not os.path.exists(training_dir):
        print("❌ 找不到 model_training 目錄")
        return False
    
    # 運行訓練腳本
    try:
        result = subprocess.run(
            [sys.executable, 'train_models.py'],
            cwd=training_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5分鐘超時
        )
        
        if result.returncode == 0:
            print("✅ 模型訓練完成!")
            print("\n📊 訓練輸出:")
            print(result.stdout)
            return True
        else:
            print("❌ 模型訓練失敗!")
            print("\n錯誤輸出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 訓練超時（超過5分鐘）")
        return False
    except Exception as e:
        print(f"❌ 運行錯誤: {str(e)}")
        return False

def check_models():
    """檢查生成的模型文件"""
    print("\n🔍 檢查生成的模型...")
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    required_files = [
        'knn_model.pkl',
        'random_forest_model.pkl', 
        'svm_model.pkl',
        'pytorch_model.pth',
        'scaler.pkl',
        'metadata.pkl'
    ]
    
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - 未找到")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ 缺少以下模型文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 所有模型文件已生成")
    return True

def main():
    """主函數"""
    print_banner()
    
    # 檢查依賴
    if not check_requirements():
        print("\n❌ 依賴檢查失敗，請先安裝所需套件")
        sys.exit(1)
    
    # 運行訓練
    if not run_training():
        print("\n❌ 模型訓練失敗")
        sys.exit(1)
    
    # 檢查模型
    if not check_models():
        print("\n❌ 模型文件檢查失敗")
        sys.exit(1)
    
    # 成功完成
    print("\n" + "=" * 60)
    print("🎉 模型訓練成功完成!")
    print("=" * 60)
    print("\n📁 生成的文件:")
    print("   - models/knn_model.pkl")
    print("   - models/random_forest_model.pkl")
    print("   - models/svm_model.pkl")
    print("   - models/pytorch_model.pth")
    print("   - models/scaler.pkl")
    print("   - models/metadata.pkl")
    print("\n🚀 下一步:")
    print("   運行前端應用: streamlit run frontend/app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
