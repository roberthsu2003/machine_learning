import os
import sys
from typing import Optional

# 將當前檔案所在目錄加入 sys.path，確保不論在本地或雲端從哪裡啟動，相對導入都能正常運作
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import joblib
from fastapi import FastAPI, HTTPException
import gradio as gr
from pydantic import BaseModel, Field

# ==========================================
# ZeroGPU 啟動相容性設定 (針對 Hugging Face 免費版限制)
# ==========================================
try:
    import spaces
except ImportError:
    # 本地環境或無安裝 spaces 時的 Dummy 替代類別
    class spaces:
        @staticmethod
        def GPU(func):
            return func

@spaces.GPU
def dummy_gpu_function():
    """此函數僅用於通過 Hugging Face ZeroGPU 啟動時的安全掃描"""
    return "GPU initialized"

# ==========================================
# 1. 載入模型與狀態管理
# ==========================================
model_path = os.path.join(current_dir, "iris_model.joblib")
MODEL_STATE = {}

def load_model_state():
    global MODEL_STATE
    if not os.path.exists(model_path):
        print("未檢測到模型檔案，正在自動執行訓練以生成 iris_model.joblib...")
        try:
            from train_save import train_and_save_model
            train_and_save_model()
        except Exception as e:
            raise RuntimeError(f"自動訓練模型失敗: {str(e)}")

    # 載入模型與相關元數據
    model_data = joblib.load(model_path)
    MODEL_STATE.clear()
    MODEL_STATE.update({
        "model": model_data["model"],
        "target_names": model_data["target_names"],
        "feature_names": model_data.get("feature_names", ["sepal length", "sepal width", "petal length", "petal width"]),
        "feature_importances": model_data.get("feature_importances", {}),
        "accuracy": model_data.get("accuracy", 0.9667),
        "train_time": model_data.get("train_time", 0.01),
        "n_estimators": model_data.get("n_estimators", 100),
        "max_depth": model_data.get("max_depth", None),
        "test_size": model_data.get("test_size", 0.2),
        "random_state": model_data.get("random_state", 42),
    })
    print("模型與類別標籤成功載入！目前準確度：", MODEL_STATE["accuracy"])

# 啟動時先載入一次狀態
load_model_state()


# ==========================================
# 2. 建立 FastAPI 應用與 Pydantic 格式定義
# ==========================================
app = FastAPI(
    title="Iris 鳶尾花機器學習服務 API",
    description="這是一個結合 FastAPI 與 Gradio 的機器學習部署服務。提供預測端點與線上訓練端點。",
    version="2.0.0",
)

# --- Pydantic 預測模型 ---
class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="花萼長度 (cm)", ge=0.1, le=10.0)
    sepal_width: float = Field(..., description="花萼寬度 (cm)", ge=0.1, le=10.0)
    petal_length: float = Field(..., description="花瓣長度 (cm)", ge=0.1, le=10.0)
    petal_width: float = Field(..., description="花瓣寬度 (cm)", ge=0.1, le=10.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    }

class IrisOutput(BaseModel):
    prediction_id: int = Field(..., description="預測類別 ID")
    prediction_label: str = Field(..., description="預測類別名稱")
    probabilities: dict[str, float] = Field(..., description="各類別預測機率")

# --- Pydantic 訓練模型 ---
class TrainConfig(BaseModel):
    n_estimators: int = Field(100, description="決策樹數量", ge=10, le=500)
    max_depth: Optional[int] = Field(None, description="最大深度 (None/0 表示無限制)", ge=0, le=20)
    test_size: float = Field(0.2, description="測試集分割比例", ge=0.1, le=0.5)
    random_state: int = Field(42, description="隨機種子", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "n_estimators": 100,
                "max_depth": 10,
                "test_size": 0.2,
                "random_state": 42
            }
        }
    }

class TrainResult(BaseModel):
    status: str = Field(..., description="執行結果狀態")
    accuracy: float = Field(..., description="測試集準確度")
    train_time: float = Field(..., description="訓練耗時 (秒)")
    feature_importances: dict[str, float] = Field(..., description="各特徵之重要性分佈")
    message: str = Field(..., description="提示訊息")


# --- FastAPI 路由端點 ---

@app.post("/predict", response_model=IrisOutput)
def predict_api(payload: IrisInput):
    """
    預測端點：接收鳶尾花的 4 項特徵，並回傳模型預測的類別與機率分佈。
    """
    features = [
        [
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]
    ]
    try:
        model = MODEL_STATE["model"]
        target_names = MODEL_STATE["target_names"]

        pred_id = int(model.predict(features)[0])
        pred_label = target_names[pred_id]

        probs = model.predict_proba(features)[0]
        prob_dict = {target_names[i]: float(p) for i, p in enumerate(probs)}

        return IrisOutput(
            prediction_id=pred_id,
            prediction_label=pred_label,
            probabilities=prob_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")


@app.post("/train", response_model=TrainResult)
def train_api(config: TrainConfig):
    """
    訓練端點：傳入決策樹數量、最大深度、測試集比例等超參數，線上重新訓練模型，並即時更新服務所使用的模型。
    """
    try:
        from train_save import train_and_save_model
        
        # 轉換 max_depth 為實體值 (0 代表 None)
        depth_val = None if config.max_depth == 0 or config.max_depth is None else config.max_depth
        
        res = train_and_save_model(
            n_estimators=config.n_estimators,
            max_depth=depth_val,
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # 線上重新載入最新模型狀態
        load_model_state()
        return TrainResult(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"線上訓練失敗: {str(e)}")


# ==========================================
# 3. 建立 Gradio UI 網頁介面 (Web UI)
# ==========================================

# --- 輔助 HTML 生成函數 ---

def make_prediction_card(label: str, prob: float) -> str:
    color_map = {
        "setosa": ("#e6f4ea", "#137333", "🌿 Setosa (山鳶尾)"),
        "versicolor": ("#fef7e0", "#b06000", "🍁 Versicolor (變色鳶尾)"),
        "virginica": ("#fce8e6", "#c5221f", "🪻 Virginica (維吉尼亞鳶尾)")
    }
    bg, fg, name = color_map.get(label, ("#f8f9fa", "#212529", label))
    return f"""
    <div style="background-color: {bg}; color: {fg}; padding: 22px; border-radius: 12px; border: 1.5px solid {fg}40; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); transition: all 0.3s ease;">
        <span style="font-size: 0.95rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85;">預測分析品種</span>
        <h2 style="font-size: 2.4rem; margin: 8px 0; font-weight: 800; letter-spacing: 0.5px;">{name}</h2>
        <span style="font-size: 1.1rem; font-weight: 500;">預測機率: <strong style="font-size: 1.5rem;">{prob:.1f}%</strong></span>
    </div>
    """

def make_probability_bars(prob_dict: dict[str, float]) -> str:
    color_scheme = {
        "setosa": "#137333",
        "versicolor": "#b06000",
        "virginica": "#c5221f"
    }
    html = '<div style="margin-top: 10px; display: flex; flex-direction: column; gap: 14px;">'
    for cls, val in prob_dict.items():
        pct = val * 100
        color = color_scheme.get(cls, "#0dcaf0")
        html += f"""
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: 600; font-size: 0.95rem;">
                <span style="text-transform: capitalize;">{cls}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background-color: #f1f3f4; border-radius: 8px; height: 12px; overflow: hidden; width: 100%;">
                <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 8px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
        </div>
        """
    html += '</div>'
    return html

def make_metrics_card(accuracy: float, train_time: float, n_est, m_depth, t_size) -> str:
    m_depth_str = "無限制" if m_depth is None or m_depth == 0 else str(m_depth)
    return f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px;">
        <div style="background-color: #f8f9fa; padding: 18px 10px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 0.8rem; color: #5f6368; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">測試集準確度</div>
            <div style="font-size: 2rem; font-weight: 800; color: #1a73e8; margin-top: 5px;">{accuracy * 100:.2f}%</div>
        </div>
        <div style="background-color: #f8f9fa; padding: 18px 10px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 0.8rem; color: #5f6368; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">模型訓練耗時</div>
            <div style="font-size: 2rem; font-weight: 800; color: #137333; margin-top: 5px;">{train_time:.4f}s</div>
        </div>
        <div style="background-color: #f8f9fa; padding: 18px 10px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 0.8rem; color: #5f6368; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">決策樹數量</div>
            <div style="font-size: 2rem; font-weight: 800; color: #ab47bc; margin-top: 5px;">{n_est}</div>
        </div>
    </div>
    <div style="font-size: 0.92rem; color: #3c4043; background: #f1f3f4; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; border: 1px solid #e0e0e0; font-weight: 500;">
        <span>🌲 <strong>最大樹深度:</strong> {m_depth_str}</span>
        <span>📊 <strong>測試集比例:</strong> {t_size * 100:.0f}%</span>
    </div>
    """

def make_importance_chart(importance_dict: dict[str, float]) -> str:
    if not importance_dict:
        return "<p style='color: #5f6368; text-align: center; padding: 20px;'>目前無特徵重要性資料</p>"
    
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    max_val = max(importance_dict.values()) if importance_dict else 1.0
    colors = ["#1a73e8", "#ab47bc", "#137333", "#e37400"]
    
    html = '<div style="margin-top: 15px; display: flex; flex-direction: column; gap: 12px;">'
    html += '<h4 style="margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700; color: #202124; letter-spacing: 0.3px;">💡 特徵重要性分析 (Feature Importance)</h4>'
    
    for idx, (feature, val) in enumerate(sorted_imp):
        pct = (val / max_val) * 100
        color = colors[idx % len(colors)]
        html += f"""
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-weight: 600; font-size: 0.9rem; color: #3c4043;">
                <span style="text-transform: capitalize;">{feature}</span>
                <span>{val * 100:.1f}%</span>
            </div>
            <div style="background-color: #f1f3f4; border-radius: 6px; height: 10px; width: 100%;">
                <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 6px; transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
        </div>
        """
    html += '</div>'
    return html


# --- Gradio 事件處理器 ---

def predict_gradio_handler(sepal_len, sepal_wid, petal_len, petal_wid):
    """
    處理 Gradio UI 的預測請求。
    """
    features = [[sepal_len, sepal_wid, petal_len, petal_wid]]
    
    model = MODEL_STATE["model"]
    target_names = MODEL_STATE["target_names"]
    
    pred_id = int(model.predict(features)[0])
    pred_label = target_names[pred_id]
    
    probs = model.predict_proba(features)[0]
    prob_dict = {target_names[i]: float(p) for i, p in enumerate(probs)}
    
    card_html = make_prediction_card(pred_label, prob_dict[pred_label] * 100)
    bars_html = make_probability_bars(prob_dict)
    
    return card_html, bars_html


def train_gradio_handler(n_estimators, max_depth, test_size, random_state):
    """
    處理 Gradio UI 的重新訓練請求。
    """
    from train_save import train_and_save_model
    
    depth_val = None if max_depth == 0 else int(max_depth)
    
    res = train_and_save_model(
        n_estimators=int(n_estimators),
        max_depth=depth_val,
        test_size=float(test_size),
        random_state=int(random_state)
    )
    
    # 重新載入全域模型狀態
    load_model_state()
    
    # 重新渲染 UI 區塊
    metrics_html = make_metrics_card(
        accuracy=MODEL_STATE["accuracy"],
        train_time=MODEL_STATE["train_time"],
        n_est=MODEL_STATE["n_estimators"],
        m_depth=MODEL_STATE["max_depth"],
        t_size=MODEL_STATE["test_size"]
    )
    importance_html = make_importance_chart(MODEL_STATE["feature_importances"])
    status_text = "### 📢 最新狀態: `✅ 線上重新訓練並載入成功！`"
    
    return status_text, metrics_html, importance_html


# --- 初始 UI 內容計算 ---
initial_pred_card, initial_pred_bars = predict_gradio_handler(5.1, 3.5, 1.4, 0.2)
initial_metrics = make_metrics_card(
    accuracy=MODEL_STATE["accuracy"],
    train_time=MODEL_STATE["train_time"],
    n_est=MODEL_STATE["n_estimators"],
    m_depth=MODEL_STATE["max_depth"],
    t_size=MODEL_STATE["test_size"]
)
initial_importance = make_importance_chart(MODEL_STATE["feature_importances"])


# --- 建立 Gradio UI Blocks 布局 ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo"),
    title="🌸 Iris 鳶尾花機器學習全生命週期平台"
) as demo:
    
    gr.Markdown(
        """
        # 🌸 Iris 鳶尾花機器學習全生命週期平台
        本系統展示了機器學習模型部署的**完整生命週期**。此服務底層使用 **FastAPI** 驅動，提供標準化 RESTful API，並結合 **Gradio** 開發了互動式 Web 介面。
        * 🔮 **即時預測分頁**：輸入鳶尾花的測量特徵，即時透過分類模型取得品種預測及信心度。
        * ⚙️ **線上訓練分頁**：可線上調整超參數，即時呼叫後端訓練模型並查看評估結果與特徵重要性。
        """
    )
    
    with gr.Tabs():
        
        # --- 分頁一：即時預測 ---
        with gr.Tab("🔮 即時模型預測"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 輸入特徵滑桿 (Features)")
                    sepal_len = gr.Slider(minimum=0.1, maximum=10.0, value=5.1, step=0.1, label="花萼長度 Sepal Length (cm)")
                    sepal_wid = gr.Slider(minimum=0.1, maximum=10.0, value=3.5, step=0.1, label="花萼寬度 Sepal Width (cm)")
                    petal_len = gr.Slider(minimum=0.1, maximum=10.0, value=1.4, step=0.1, label="花瓣長度 Petal Length (cm)")
                    petal_wid = gr.Slider(minimum=0.1, maximum=10.0, value=0.2, step=0.1, label="花瓣寬度 Petal Width (cm)")
                    
                    predict_btn = gr.Button("🔮 開始預測", variant="primary")
                    
                with gr.Column(scale=1.2):
                    gr.Markdown("### 2. 預測結果與概率分析")
                    output_card = gr.HTML(value=initial_pred_card, label="品種預測卡片")
                    output_probs = gr.HTML(value=initial_pred_bars, label="機率分析")
            
            # 綁定即時變更事件 (Slider 變動時即時進行預測)
            inputs = [sepal_len, sepal_wid, petal_len, petal_wid]
            outputs = [output_card, output_probs]
            for slider in inputs:
                slider.change(fn=predict_gradio_handler, inputs=inputs, outputs=outputs)
            predict_btn.click(fn=predict_gradio_handler, inputs=inputs, outputs=outputs)
            
        # --- 分頁二：線上訓練 ---
        with gr.Tab("⚙️ 線上模型訓練與評估"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 調整隨機森林超參數")
                    n_est = gr.Slider(minimum=10, maximum=500, value=MODEL_STATE["n_estimators"], step=10, label="決策樹樹量 (n_estimators)")
                    m_depth = gr.Slider(minimum=0, maximum=20, value=MODEL_STATE["max_depth"] if MODEL_STATE["max_depth"] is not None else 0, step=1, label="最大深度 (max_depth) - 設為 0 表示無限制")
                    t_size = gr.Slider(minimum=0.1, maximum=0.5, value=MODEL_STATE["test_size"], step=0.05, label="測試集比例 (test_size)")
                    seed = gr.Number(value=MODEL_STATE["random_state"], label="隨機種子 (random_state)", precision=0)
                    
                    train_btn = gr.Button("🚀 開始訓練模型", variant="primary")
                    
                with gr.Column(scale=1.2):
                    gr.Markdown("### 2. 訓練結果與特徵重要性")
                    train_status = gr.Markdown("### 📢 最新狀態: `已載入預訓練模型 (就緒)`")
                    metrics_card = gr.HTML(value=initial_metrics, label="評估指標卡片")
                    importance_chart = gr.HTML(value=initial_importance, label="特徵重要性圖表")
            
            # 綁定訓練按鈕事件
            train_btn.click(
                fn=train_gradio_handler,
                inputs=[n_est, m_depth, t_size, seed],
                outputs=[train_status, metrics_card, importance_chart]
            )

# ==========================================
# 4. 將 Gradio UI 掛載到 FastAPI 上
# ==========================================
# 當用戶訪問 "/" 時會自動導向 Gradio Web UI，而對外則繼續提供 API 端點
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    
    # 偵測是否運行於 Hugging Face Spaces 環境
    is_hf = os.environ.get("SYSTEM") == "spaces"
    
    host = "0.0.0.0" if is_hf else "127.0.0.1"
    port = int(os.environ.get("PORT", 7860)) if is_hf else 8000
    reload_mode = False if is_hf else True
    
    print(f"啟動服務中... (Host: {host}, Port: {port}, Reload: {reload_mode})")
    uvicorn.run("app:app", host=host, port=port, reload=reload_mode)
