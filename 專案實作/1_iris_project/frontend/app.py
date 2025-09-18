"""
Streamlit 前端應用程式
提供互動式的機器學習模型展示和預測功能
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# 導入自定義模組
from model_loader import load_models, get_model_loader

# 設定頁面配置
st.set_page_config(
    page_title="🤖 機器學習模型展示平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1e88e5;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #424242;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 3px solid #1e88e5;
        padding-bottom: 0.5rem;
    }
    .highlight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #1e88e5, #42a5f5);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """初始化應用程式"""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    if not st.session_state.models_loaded:
        with st.spinner("🔄 載入模型中..."):
            if load_models():
                st.session_state.models_loaded = True
                st.success("✅ 模型載入成功!")
            else:
                st.error("❌ 模型載入失敗! 請確保已運行模型訓練階段。")
                st.stop()

def create_header():
    """創建頁面標題"""
    st.markdown('<h1 class="main-header">🤖 機器學習模型展示平台</h1>', unsafe_allow_html=True)
    
    # 添加描述
    st.markdown("""
    <div class="highlight-box">
    <h3>🎯 平台特色</h3>
    <ul>
    <li><strong>多模型支援</strong>：同時使用多個訓練好的模型進行預測</li>
    <li><strong>互動式介面</strong>：滑桿輸入、即時預測、視覺化展示</li>
    <li><strong>模型比較</strong>：並排比較不同模型的預測結果和性能</li>
    <li><strong>響應式設計</strong>：適配不同螢幕尺寸的現代化介面</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """創建側邊欄控制面板"""
    st.sidebar.title("🎛️ 控制面板")
    
    # 獲取模型載入器
    loader = get_model_loader()
    
    # 模型選擇
    available_models = loader.get_available_models()
    if not available_models:
        st.sidebar.error("❌ 沒有可用的模型")
        return None, None, None, None, None
    
    selected_models = st.sidebar.multiselect(
        "🔧 選擇要使用的模型",
        available_models,
        default=available_models,
        format_func=lambda x: loader.model_configs[x]['name']
    )
    
    if not selected_models:
        st.sidebar.warning("⚠️ 請至少選擇一個模型")
        return None, None, None, None, None
    
    # 輸入控制
    st.sidebar.subheader("🌸 輸入花朵測量值")
    
    feature_names = loader.get_feature_names()
    
    # 創建滑桿輸入
    inputs = {}
    for i, feature_name in enumerate(feature_names):
        if '長度' in feature_name:
            min_val, max_val, default_val = 4.0, 8.0, 5.1
        elif '寬度' in feature_name and '萼片' in feature_name:
            min_val, max_val, default_val = 2.0, 5.0, 3.5
        elif '寬度' in feature_name and '花瓣' in feature_name:
            min_val, max_val, default_val = 0.1, 3.0, 0.2
        else:  # 花瓣長度
            min_val, max_val, default_val = 1.0, 7.0, 1.4
        
        inputs[feature_name] = st.sidebar.slider(
            feature_name,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.1,
            help=f"調整 {feature_name} 的測量值"
        )
    
    # 預設值按鈕
    st.sidebar.subheader("📋 快速輸入")
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("🌺 山鳶尾範例"):
        return selected_models, [5.1, 3.5, 1.4, 0.2], feature_names, loader, True
    
    if col2.button("🌸 變色鳶尾範例"):
        return selected_models, [6.2, 2.9, 4.3, 1.3], feature_names, loader, True
    
    if st.sidebar.button("🌹 維吉尼亞鳶尾範例"):
        return selected_models, [6.5, 3.0, 5.2, 2.0], feature_names, loader, True
    
    # 返回當前輸入值
    input_values = [inputs[name] for name in feature_names]
    return selected_models, input_values, feature_names, loader, False

def display_predictions(selected_models, input_values, feature_names, loader):
    """顯示預測結果"""
    st.markdown('<h2 class="section-header">🎯 預測結果</h2>', unsafe_allow_html=True)
    
    # 準備輸入數據
    X_new = np.array([input_values])
    
    # 進行預測
    with st.spinner("🔮 正在預測..."):
        results = loader.predict_all_models(X_new)
    
    # 顯示預測結果
    target_names = loader.get_target_names()
    
    # 創建預測結果卡片
    cols = st.columns(len(selected_models))
    predictions = {}
    
    for i, model_name in enumerate(selected_models):
        with cols[i]:
            if model_name in results and results[model_name]['predictions'] is not None:
                pred_class_idx = results[model_name]['predictions'][0]
                pred_class = target_names[pred_class_idx]
                predictions[model_name] = pred_class_idx
                
                st.markdown(f"""
                <div class="metric-card">
                <h4>{results[model_name]['model_name']}</h4>
                <h2 style="color: #1e88e5;">{pred_class}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ {model_name} 預測失敗")
    
    # 顯示輸入特徵值
    st.markdown("### 📊 輸入特徵值")
    feature_df = pd.DataFrame({
        '特徵': feature_names,
        '數值': input_values
    })
    st.dataframe(feature_df, width='stretch')
    
    return predictions, results

def display_probabilities(results, selected_models, target_names):
    """顯示預測機率分佈"""
    # 收集有機率信息的模型
    prob_data = []
    for model_name in selected_models:
        if (model_name in results and 
            results[model_name]['probabilities'] is not None):
            
            for i, target_name in enumerate(target_names):
                prob_data.append({
                    '模型': results[model_name]['model_name'],
                    '品種': target_name,
                    '機率': results[model_name]['probabilities'][0][i]
                })
    
    if prob_data:
        st.markdown("### 📈 預測機率分佈")
        
        prob_df = pd.DataFrame(prob_data)
        
        # 創建機率比較圖
        fig = px.bar(
            prob_df,
            x='品種',
            y='機率',
            color='模型',
            title="各模型預測機率比較",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title="鳶尾花品種",
            yaxis_title="預測機率",
            height=500
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # 顯示數值表格
        st.markdown("#### 📋 機率數值表")
        pivot_df = prob_df.pivot(index='品種', columns='模型', values='機率')
        st.dataframe(pivot_df.round(3), width='stretch')

def analyze_model_consensus(predictions, loader):
    """分析模型一致性"""
    if len(predictions) < 2:
        return
    
    st.markdown("### 🤝 模型一致性分析")
    
    # 分析預測一致性
    unique_predictions = set(predictions.values())
    target_names = loader.get_target_names()
    
    if len(unique_predictions) == 1:
        pred_name = target_names[list(unique_predictions)[0]]
        st.markdown(f"""
        <div class="prediction-box">
        <h4>✅ 模型預測一致</h4>
        <p>所有選中的模型都預測為：<strong>{pred_name}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ 模型預測存在分歧</h4>
        <p>不同模型給出了不同的預測結果，這可能表示：</p>
        <ul>
        <li>輸入數據位於不同類別的邊界區域</li>
        <li>不同模型對特徵的敏感度不同</li>
        <li>可以考慮使用集成學習方法</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 顯示分歧詳情
        pred_groups = {}
        for model_name, pred_idx in predictions.items():
            pred_name = target_names[pred_idx]
            if pred_name not in pred_groups:
                pred_groups[pred_name] = []
            pred_groups[pred_name].append(loader.model_configs[model_name]['name'])
        
        for pred_name, model_list in pred_groups.items():
            st.write(f"**{pred_name}**: {', '.join(model_list)}")

def display_feature_space_visualization(input_values, feature_names, loader):
    """顯示特徵空間視覺化"""
    st.markdown("### 📍 在特徵空間中的位置")
    
    # 這裡可以添加特徵空間的視覺化
    # 例如：2D 散點圖顯示預測點的位置
    
    # 創建一個簡單的特徵值條形圖
    fig = px.bar(
        x=feature_names,
        y=input_values,
        title="輸入特徵值分佈",
        color=input_values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="特徵名稱",
        yaxis_title="特徵值",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')

def display_model_info(loader):
    """顯示模型資訊"""
    st.markdown('<h2 class="section-header">📚 模型資訊</h2>', unsafe_allow_html=True)
    
    # 獲取模型資訊
    model_info = loader.get_model_info()
    data_summary = loader.get_data_summary()
    
    # 顯示數據集資訊
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("樣本總數", data_summary['n_samples'])
    with col2:
        st.metric("特徵數量", data_summary['n_features'])
    with col3:
        st.metric("分類種類", data_summary['n_classes'])
    with col4:
        st.metric("可用模型", len(model_info))
    
    # 顯示模型詳情
    st.markdown("#### 🔧 可用模型")
    
    for model_key, info in model_info.items():
        with st.expander(f"{info['name']} ({info['type']})"):
            if info['type'] == 'traditional':
                st.write("**類型**: 傳統機器學習模型")
                st.write("**特點**: 快速、可解釋性強")
            else:
                st.write("**類型**: 深度學習模型")
                st.write("**特點**: 複雜模式識別、可擴展性強")
            
            st.write(f"**狀態**: {'✅ 已載入' if info['available'] else '❌ 未載入'}")

def main():
    """主應用程式函數"""
    # 初始化應用程式
    initialize_app()
    
    # 創建頁面標題
    create_header()
    
    # 創建側邊欄
    sidebar_result = create_sidebar()
    if sidebar_result[0] is None:
        st.warning("⚠️ 請在側邊欄中選擇模型和輸入參數")
        return
    
    selected_models, input_values, feature_names, loader, is_example = sidebar_result
    
    # 顯示預測結果
    predictions, results = display_predictions(selected_models, input_values, feature_names, loader)
    
    # 顯示機率分佈
    target_names = loader.get_target_names()
    display_probabilities(results, selected_models, target_names)
    
    # 分析模型一致性
    analyze_model_consensus(predictions, loader)
    
    # 顯示特徵空間視覺化
    display_feature_space_visualization(input_values, feature_names, loader)
    
    # 顯示模型資訊
    display_model_info(loader)
    
    # 頁腳
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 機器學習模型展示平台 | 基於 Streamlit 和 PyTorch 構建</p>
        <p>📚 學習目標：掌握機器學習模型的實際應用和比較分析</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
