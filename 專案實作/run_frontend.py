#!/usr/bin/env python3
"""
一鍵運行前端應用腳本
簡化前端啟動流程
"""

import os
import sys
import subprocess
import time

def print_banner():
    """打印歡迎橫幅"""
    print("=" * 60)
    print("🎨 機器學習前端應用啟動器")
    print("=" * 60)
    print()

def check_models():
    """檢查模型文件是否存在"""
    print("🔍 檢查模型文件...")
    
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
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 未找到")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ 缺少以下模型文件: {', '.join(missing_files)}")
        print("請先運行: python run_training.py")
        return False
    
    print("✅ 所有模型文件已就緒")
    return True

def check_frontend_requirements():
    """檢查前端依賴"""
    print("\n🔍 檢查前端依賴...")
    
    required_packages = [
        'streamlit', 'plotly', 'numpy', 'pandas', 
        'torch', 'scikit-learn', 'joblib'
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
    
    print("✅ 前端依賴已就緒")
    return True

def run_frontend():
    """運行前端應用"""
    print("\n🚀 啟動前端應用...")
    print("-" * 40)
    
    # 檢查前端目錄
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    app_file = os.path.join(frontend_dir, 'app.py')
    
    if not os.path.exists(app_file):
        print("❌ 找不到 frontend/app.py")
        return False
    
    print("🌐 正在啟動 Streamlit 應用...")
    print("📱 應用將在瀏覽器中自動開啟")
    print("🔗 如果沒有自動開啟，請訪問: http://localhost:8501")
    print("\n按 Ctrl+C 停止應用")
    print("-" * 40)
    
    try:
        # 運行 Streamlit 應用
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0'
        ], cwd=frontend_dir)
        
    except KeyboardInterrupt:
        print("\n\n👋 應用已停止")
        return True
    except Exception as e:
        print(f"\n❌ 運行錯誤: {str(e)}")
        return False

def show_usage_tips():
    """顯示使用提示"""
    print("\n💡 使用提示:")
    print("1. 在側邊欄調整花朵測量值")
    print("2. 選擇要使用的模型")
    print("3. 點擊「🔮 開始預測」查看結果")
    print("4. 觀察不同模型的預測比較")
    print("5. 查看機率分佈和一致性分析")

def main():
    """主函數"""
    print_banner()
    
    # 檢查模型文件
    if not check_models():
        print("\n❌ 模型文件檢查失敗")
        print("請先運行: python run_training.py")
        sys.exit(1)
    
    # 檢查前端依賴
    if not check_frontend_requirements():
        print("\n❌ 前端依賴檢查失敗")
        sys.exit(1)
    
    # 顯示使用提示
    show_usage_tips()
    
    # 等待用戶確認
    input("\n按 Enter 鍵繼續啟動應用...")
    
    # 運行前端
    if not run_frontend():
        print("\n❌ 前端應用啟動失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()
