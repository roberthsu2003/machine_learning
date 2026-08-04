# 必需配合,practice_predict_api_answer.ipynb檔,才可以完全了解

import os
import sys
from typing import Optional

# 將當前檔案所在目錄加入 sys.path，確保不論在本地或雲端從哪裡啟動，相對導入都能正常運作
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from train_save import train_and_save_model  # type: ignore
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
import gradio as gr
from pydantic import BaseModel, Field

# ==========================================
# 1. 載入模型與狀態管理
# ==========================================
model_path = os.path.join(current_dir, "salary_model.joblib")
MODEL_STATE = {}

def load_model_state():
    global MODEL_STATE
    if not os.path.exists(model_path):
        print("未檢測到模型檔案，正在自動執行訓練以生成 salary_model.joblib...")
        try:
            train_and_save_model()
        except Exception as e:
            raise RuntimeError(f"自動訓練模型失敗: {str(e)}")

    # 載入模型與相關元數據
    model_data = joblib.load(model_path)
    MODEL_STATE.clear()
    MODEL_STATE.update({
        "model": model_data["model"],
        "oe": model_data["oe"],
        "ohe": model_data["ohe"],
        "scaler": model_data["scaler"],
        "r2": model_data.get("r2", 0.8463),
        "coef": model_data.get("coef", []),
        "intercept": model_data.get("intercept", 51.2286),
        "feature_names": model_data.get("feature_names", ['YearsExperience', 'EducationLevel', 'City_城市A', 'City_城市B', 'City_城市C']),
        "feature_coefs": model_data.get("feature_coefs", {}),
        "model_type": model_data.get("model_type", "LinearRegression"),
        "alpha": model_data.get("alpha", 1.0),
        "train_time": model_data.get("train_time", 0.01),
        "test_size": model_data.get("test_size", 0.2),
        "random_state": model_data.get("random_state", 76),
    })
    print("模型與預處理器成功載入！目前 R² Score：", MODEL_STATE["r2"])

# 啟動時先載入一次狀態
load_model_state()


# ==========================================
# 2. 建立 FastAPI 應用與 Pydantic 格式定義
# ==========================================
api_app = FastAPI(
    title="薪資預測多元線性迴歸 API",
    description="這是一個結合 FastAPI 與 Gradio 的機器學習部署服務。提供薪資預測端點與線上模型訓練端點。",
    version="2.0.0",
)

# --- Pydantic 預測模型 ---
class SalaryInput(BaseModel):
    years_experience: float = Field(..., description="工作年資 (年，通常為 1.0 ~ 10.0)", ge=0.0, le=50.0)
    education_level: str = Field(..., description="學歷 (大學、碩士以上、高中以下)")
    city: str = Field(..., description="工作城市 (城市A、城市B、城市C)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "years_experience": 5.3,
                "education_level": "碩士以上",
                "city": "城市A"
            }
        }
    }

class SalaryOutput(BaseModel):
    predicted_salary: float = Field(..., description="預測月薪 (k / 千元)")
    estimated_annual_salary: float = Field(..., description="估計年薪 (k / 千元，以 14 個月估算)")

# --- Pydantic 訓練模型 ---
class TrainConfig(BaseModel):
    test_size: float = Field(0.2, description="測試集分割比例", ge=0.1, le=0.5)
    random_state: int = Field(76, description="隨機種子", ge=0)
    model_type: str = Field("LinearRegression", description="模型演算法類型 (LinearRegression, Lasso, Ridge)")
    alpha: float = Field(1.0, description="正則化強度 alpha (適用於 Lasso 與 Ridge)", ge=0.001, le=100.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "test_size": 0.2,
                "random_state": 76,
                "model_type": "LinearRegression",
                "alpha": 1.0
            }
        }
    }

class TrainResult(BaseModel):
    status: str = Field(..., description="執行結果狀態")
    r2: float = Field(..., description="測試集 R-squared 決定係數")
    coef: list[float] = Field(..., description="特徵權重係數列表")
    intercept: float = Field(..., description="截距")
    feature_coefs: dict[str, float] = Field(..., description="特徵及其權重映射")
    model_type: str = Field(..., description="模型演算法類型")
    alpha: float = Field(..., description="正則化強度 alpha")
    train_time: float = Field(..., description="訓練耗時 (秒)")
    message: str = Field(..., description="提示訊息")


# --- FastAPI 路由端點 ---

@api_app.post("/predict", response_model=SalaryOutput)
def predict_api(payload: SalaryInput):
    """
    預測端點：接收年資、學歷、城市，進行編碼與標準化後，回傳模型預測的月薪與估計年薪。
    """
    try:
        oe = MODEL_STATE["oe"]
        ohe = MODEL_STATE["ohe"]
        scaler = MODEL_STATE["scaler"]
        model = MODEL_STATE["model"]

        # 1. 學歷編碼 (使用 OrdinalEncoder 順序編碼)
        try:
            edu_encoded = int(oe.transform(pd.DataFrame([[payload.education_level]], columns=["EducationLevel"]))[0][0])
        except ValueError:
            valid_cats = list(oe.categories_[0])
            raise HTTPException(
                status_code=400, 
                detail=f"未知的學歷: {payload.education_level}。可接受的值為: {valid_cats}"
            )

        # 2. 城市獨熱編碼
        try:
            city_encoded = ohe.transform(pd.DataFrame([[payload.city]], columns=["City"]))[0]
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"未知的城市: {payload.city}。可接受的值為: 城市A, 城市B, 城市C"
            )

        # 3. 拼接特徵 (順序必須為：YearsExperience, EducationLevel, City_城市A, City_城市B, City_城市C)
        feature_names = MODEL_STATE["feature_names"]
        features_df = pd.DataFrame([[payload.years_experience, edu_encoded] + list(city_encoded)], columns=feature_names)

        # 4. 標準化
        features_scaled = scaler.transform(features_df)

        # 5. 進行預測
        pred_val = float(model.predict(features_scaled)[0])

        return SalaryOutput(
            predicted_salary=pred_val,
            estimated_annual_salary=pred_val * 14
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")


@api_app.post("/train", response_model=TrainResult)
def train_api(config: TrainConfig):
    """
    訓練端點：傳入測試集比例、隨機種子、模型類型與 alpha，線上重新訓練模型，並即時更新服務所使用的模型。
    """
    try:
        res = train_and_save_model(
            test_size=config.test_size,
            random_state=config.random_state,
            model_type=config.model_type,
            alpha=config.alpha
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

def make_prediction_card(salary: float) -> str:
    annual = salary * 14
    return f"""
    <div style="background: linear-gradient(135deg, #0f9b0f, #38ef7d); color: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 20px; transition: all 0.3s ease;">
        <span style="font-size: 0.95rem; font-weight: bold; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.95;">預測月薪薪資</span>
        <h2 style="font-size: 2.8rem; margin: 10px 0; font-weight: 800; text-shadow: 1px 1px 3px rgba(0,0,0,0.15);">{salary:.2f} <span style="font-size: 1.5rem; font-weight: 400;">k</span></h2>
        <span style="font-size: 1.05rem; font-weight: 500; opacity: 0.9;">估計年薪 (14個月): <strong style="font-size: 1.3rem;">{annual:.1f}</strong> k</span>
    </div>
    """

def make_metrics_card(r2: float, train_time: float, intercept: float, test_size: float, random_state: int, model_type: str = "LinearRegression", alpha: float = 1.0) -> str:
    alpha_info = f" (α={alpha})" if model_type.lower() in ["lasso", "ridge"] else ""
    return f"""
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px;">
        <div style="background-color: #f8f9fa; padding: 18px 10px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 0.8rem; color: #5f6368; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">決定係數 R² Score</div>
            <div style="font-size: 2rem; font-weight: 800; color: #1a73e8; margin-top: 5px;">{r2:.4f}</div>
        </div>
        <div style="background-color: #f8f9fa; padding: 18px 10px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 0.8rem; color: #5f6368; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">模型訓練耗時</div>
            <div style="font-size: 2rem; font-weight: 800; color: #137333; margin-top: 5px;">{train_time:.4f}s</div>
        </div>
    </div>
    <div style="font-size: 0.92rem; color: #3c4043; background: #e8f0fe; padding: 12px 18px; border-radius: 8px; border: 1px solid #d2e3fc; font-weight: 600; margin-bottom: 10px;">
        🤖 <strong>模型演算法:</strong> <span style="color: #1a73e8;">{model_type}{alpha_info}</span>
    </div>
    <div style="font-size: 0.92rem; color: #3c4043; background: #f1f3f4; padding: 12px 18px; border-radius: 8px; border: 1px solid #e0e0e0; font-weight: 500; margin-bottom: 10px;">
        🏠 <strong>模型截距 (Intercept / 偏置值 b):</strong> {intercept:.4f}
    </div>
    <div style="font-size: 0.85rem; color: #5f6368; display: flex; justify-content: space-between; font-weight: 500; padding: 0 5px;">
        <span>📊 <strong>測試集比例:</strong> {test_size * 100:.0f}%</span>
        <span>🌱 <strong>隨機種子:</strong> {random_state}</span>
    </div>
    """

def make_equation_html(feature_coefs: dict[str, float], intercept: float) -> str:
    html_parts = []
    for name, coef in feature_coefs.items():
        color = "#137333" if coef >= 0 else "#c5221f"
        sign = "+" if coef >= 0 else "-"
        html_parts.append(f"""
        <span style="white-space: nowrap; margin: 0 4px; display: inline-block;">
            {sign} <strong style="color: {color};">{abs(coef):.3f}</strong> × <span style="color: #202124; font-weight: 600;">({name})</span>
        </span>
        """)
    
    html_eq = f"""
    <div style="background-color: #f8f9fa; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 6px; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; border-left: 5px solid #1a73e8;">
        <h4 style="margin: 0 0 8px 0; color: #202124; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">🧮 擬合迴歸方程式 (Fitted Equation)</h4>
        <div style="font-family: 'Consolas', 'Courier New', Courier, monospace; font-size: 1.05rem; color: #3c4043; line-height: 1.6; word-wrap: break-word; padding: 5px 0;">
            <strong style="color: #1a73e8;">Salary (預測月薪)</strong> = 
            <span style="font-weight: bold;">{intercept:.3f}</span>
            {" ".join(html_parts)}
        </div>
        <p style="font-size: 0.78rem; color: #5f6368; margin: 8px 0 0 0; line-height: 1.45; border-top: 1px dashed #e0e0e0; padding-top: 8px;">
            *註：方程式中的變數皆為<strong>標準化 (Standardized)</strong> 後的數值。權重為正（<span style="color: #137333; font-weight: bold;">綠色</span>）代表該特徵增加會提升薪資，權重為負（<span style="color: #c5221f; font-weight: bold;">紅色</span>）代表該特徵增加會降低薪資。
        </p>
    </div>
    """
    return html_eq

def make_importance_chart(feature_coefs: dict[str, float]) -> str:
    if not feature_coefs:
        return "<p style='color: #5f6368; text-align: center; padding: 20px;'>目前無特徵權重資料</p>"
    
    # 依權重絕對值由大到小排序
    sorted_coefs = sorted(feature_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    max_abs_val = max(abs(val) for val in feature_coefs.values()) if feature_coefs else 1.0
    if max_abs_val == 0:
        max_abs_val = 1.0
        
    html = '<div style="margin-top: 15px; display: flex; flex-direction: column; gap: 14px;">'
    html += '<h4 style="margin: 0 0 8px 0; font-size: 1.1rem; font-weight: 700; color: #202124; letter-spacing: 0.3px;">💡 特徵影響力分析 (Feature Coefficients)</h4>'
    
    for feature, val in sorted_coefs:
        pct = (abs(val) / max_abs_val) * 100
        color = "#137333" if val >= 0 else "#c5221f"
        direction_text = " (正向加薪 📈)" if val >= 0 else " (負向減薪 📉)"
        html += f"""
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-weight: 600; font-size: 0.95rem; color: #3c4043;">
                <span>{feature}{direction_text}</span>
                <span style="color: {color}; font-family: monospace;">{val:+.4f}</span>
            </div>
            <div style="background-color: #f1f3f4; border-radius: 8px; height: 12px; overflow: hidden; width: 100%;">
                <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 8px; transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);"></div>
            </div>
        </div>
        """
    html += '</div>'
    return html


# --- Gradio 事件處理器 ---

def predict_gradio_handler(years_exp, edu_level, city):
    """
    處理 Gradio UI 的預測請求。
    """
    oe = MODEL_STATE["oe"]
    ohe = MODEL_STATE["ohe"]
    scaler = MODEL_STATE["scaler"]
    model = MODEL_STATE["model"]
    
    edu_encoded = int(oe.transform(pd.DataFrame([[edu_level]], columns=["EducationLevel"]))[0][0])
    city_encoded = ohe.transform(pd.DataFrame([[city]], columns=["City"]))[0]
    
    feature_names = MODEL_STATE["feature_names"]
    features_df = pd.DataFrame([[years_exp, edu_encoded] + list(city_encoded)], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    
    pred_val = float(model.predict(features_scaled)[0])
    
    card_html = make_prediction_card(pred_val)
    return card_html


def train_gradio_handler(test_size, random_state, model_type, alpha):
    """
    處理 Gradio UI 的重新訓練請求。
    """
    res = train_and_save_model(
        test_size=float(test_size),
        random_state=int(random_state),
        model_type=str(model_type),
        alpha=float(alpha)
    )
    
    # 重新載入全域模型狀態
    load_model_state()
    
    # 重新渲染 UI 區塊
    metrics_html = make_metrics_card(
        r2=MODEL_STATE["r2"],
        train_time=MODEL_STATE["train_time"],
        intercept=MODEL_STATE["intercept"],
        test_size=MODEL_STATE["test_size"],
        random_state=MODEL_STATE["random_state"],
        model_type=MODEL_STATE.get("model_type", "LinearRegression"),
        alpha=MODEL_STATE.get("alpha", 1.0)
    )
    equation_html = make_equation_html(MODEL_STATE["feature_coefs"], MODEL_STATE["intercept"])
    importance_html = make_importance_chart(MODEL_STATE["feature_coefs"])
    status_text = f"### 📢 最新狀態: `✅ {MODEL_STATE.get('model_type', 'LinearRegression')} 模型線上重新訓練並載入成功！`"
    
    return status_text, metrics_html, equation_html, importance_html


# --- 初始 UI 內容計算 ---
initial_pred_card = predict_gradio_handler(5.0, "大學", "城市A")
initial_metrics = make_metrics_card(
    r2=MODEL_STATE["r2"],
    train_time=MODEL_STATE["train_time"],
    intercept=MODEL_STATE["intercept"],
    test_size=MODEL_STATE["test_size"],
    random_state=MODEL_STATE["random_state"],
    model_type=MODEL_STATE.get("model_type", "LinearRegression"),
    alpha=MODEL_STATE.get("alpha", 1.0)
)
initial_equation = make_equation_html(MODEL_STATE["feature_coefs"], MODEL_STATE["intercept"])
initial_importance = make_importance_chart(MODEL_STATE["feature_coefs"])


# --- 建立 Gradio UI Blocks 布局 ---
with gr.Blocks(
    title="💼 薪資預測多元線性迴歸平台"
) as demo:
    
    gr.Markdown(
        """
        # 💼 薪資預測多元線性迴歸教學與部署平台
        本系統展示了機器學習模型部署的**完整生命週期**。此服務底層使用 **FastAPI** 驅動，提供標準化 RESTful API，並結合 **Gradio** 開發了互動式 Web 介面。
        * 🔮 **即時預測分頁**：輸入您的工作年資、學歷與工作城市，即時透過多元線性迴歸模型取得月薪與年薪估計。
        * ⚙️ **線上訓練與公式分頁**：可線上調整測試集切分比例與隨機種子，即時訓練模型，並動態展示擬合後的**數學迴歸方程式**與特徵權重係數。
        """
    )
    
    with gr.Tabs():
        
        # --- 分頁一：即時預測 ---
        with gr.Tab("🔮 即時月薪預測"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 輸入特徵 (Features)")
                    years_exp = gr.Slider(minimum=1.0, maximum=10.0, value=5.0, step=0.1, label="工作年資 (Years Experience)")
                    edu_level = gr.Dropdown(choices=["大學", "碩士以上", "高中以下"], value="大學", label="教育學歷 (Education Level)")
                    city = gr.Dropdown(choices=["城市A", "城市B", "城市C"], value="城市A", label="工作城市 (City)")
                    
                    predict_btn = gr.Button("🔮 開始預測", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 2. 預測結果")
                    output_card = gr.HTML(value=initial_pred_card, label="薪資預測卡片")
            
            # 針對 Render 伺服器優化：使用 .release() 代替 .change()，避免拖曳中發送大量請求
            # 針對 Dropdown，因為它沒有拖動過程，直接使用 .change() 是安全的
            inputs = [years_exp, edu_level, city]
            
            years_exp.release(
                fn=predict_gradio_handler,
                inputs=inputs,
                outputs=[output_card],
                queue=False,
                show_progress="hidden",
            )
            edu_level.change(
                fn=predict_gradio_handler,
                inputs=inputs,
                outputs=[output_card],
                queue=False,
                show_progress="hidden",
            )
            city.change(
                fn=predict_gradio_handler,
                inputs=inputs,
                outputs=[output_card],
                queue=False,
                show_progress="hidden",
            )
            predict_btn.click(
                fn=predict_gradio_handler,
                inputs=inputs,
                outputs=[output_card],
                queue=False,
                show_progress="hidden",
            )
            
        # --- 分頁二：線上訓練 ---
        with gr.Tab("⚙️ 線上模型訓練與公式解析"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 調整訓練參數")
                    m_type = gr.Dropdown(
                        choices=["LinearRegression", "Lasso", "Ridge"],
                        value=MODEL_STATE.get("model_type", "LinearRegression"),
                        label="模型演算法 (Model Type)"
                    )
                    alpha_val = gr.Slider(
                        minimum=0.01, maximum=10.0, value=MODEL_STATE.get("alpha", 1.0), step=0.05,
                        label="正則化強度 (alpha / 懲罰項，僅適用 Lasso & Ridge)"
                    )
                    t_size = gr.Slider(minimum=0.1, maximum=0.5, value=MODEL_STATE["test_size"], step=0.05, label="測試集比例 (test_size)")
                    seed = gr.Number(value=MODEL_STATE["random_state"], label="隨機種子 (random_state)", precision=0)
                    
                    train_btn = gr.Button("🚀 開始訓練模型", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 2. 訓練結果與特徵分析")
                    train_status = gr.Markdown("### 📢 最新狀態: `已載入預訓練模型 (就緒)`")
                    metrics_card = gr.HTML(value=initial_metrics, label="評估指標卡片")
                    equation_box = gr.HTML(value=initial_equation, label="迴歸方程式")
                    importance_chart = gr.HTML(value=initial_importance, label="特徵重要性圖表")
            
            # 綁定訓練按鈕事件
            train_btn.click(
                fn=train_gradio_handler,
                inputs=[t_size, seed, m_type, alpha_val],
                outputs=[train_status, metrics_card, equation_box, importance_chart],
                queue=False,
                show_progress="minimal",
            )

# 設定主題（避免 Gradio 6.0 的 Blocks 建構警告）
demo.theme = gr.themes.Soft(primary_hue="teal", secondary_hue="indigo")

# ⚠️ 指派 demo.theme 之後，手動計算主題的 CSS 與雜湊值，防止直接用 uvicorn 啟動時發生 500 錯誤
import hashlib
demo.theme_css = demo.theme._get_theme_css()
demo.stylesheets = demo.theme._stylesheets
demo.theme_hash = hashlib.sha256(demo.theme_css.encode("utf-8")).hexdigest()

# 設定佇列上限，防止連續操作時阻塞
demo.queue(default_concurrency_limit=10)

# ==========================================
# 4. 融合 Gradio 與自訂 API 路由
# ==========================================

# 1. 產生 Gradio 的 FastAPI 應用實例
app = gr.routes.App.create_app(demo)

# 2. 合併 API 路由：將 api_app 中的所有自訂 API 路由 (/predict, /train) 併入
app.include_router(api_app.router)

# 3. 顯式註冊被 Gradio 萬用路由隱藏的 Swagger UI 與 openapi.json
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Salary Prediction API - Swagger UI"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    return app.openapi()


if __name__ == "__main__":
    import uvicorn
    # Render 會透過 PORT 環境變數指定對外埠號；本地開發預設 8000
    port = int(os.environ.get("PORT", 8000))
    # 本地開發可設定環境變數 RELOAD=true 啟用熱重載；Render 生產環境維持關閉
    reload = os.environ.get("RELOAD", "").lower() == "true"
    print(f"使用 uvicorn 啟動伺服器 (port={port}, reload={reload})...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
