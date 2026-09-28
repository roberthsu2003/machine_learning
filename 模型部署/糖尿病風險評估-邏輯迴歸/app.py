import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

from train_save import train_and_save_model

# 1. 載入模型元數據
MODEL_PATH = "diabetes_model.joblib"
if not os.path.exists(MODEL_PATH):
    train_and_save_model(model_path=MODEL_PATH)

meta = joblib.load(MODEL_PATH)
model = meta["model"]
scaler = meta["scaler"]
odds_ratios = meta["odds_ratios"]

# 中文字型設定
font_path = "ChineseFont.ttf" if os.path.exists("ChineseFont.ttf") else "../../source_data/ChineseFont.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
    plt.rcParams["axes.unicode_minus"] = False
else:
    font_prop = None

# 2. 建立 FastAPI 實例
app = FastAPI(
    title="糖尿病患病風險評估 API (Logistic Regression)",
    description="結合 FastAPI 與 Gradio 的智慧醫療風險評估與線上模型熱重載系統",
    version="1.0.0"
)

# 3. Pydantic 資料模型
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, description="病患年齡 (歲)", example=45)
    weight: float = Field(..., ge=20, le=250, description="病患體重 (公斤)", example=75.5)
    blood_sugar: float = Field(..., ge=40, le=400, description="空腹血糖 (mg/dL)", example=126.0)
    gender: str = Field(..., description="生理性別 ('男生' 或 '女生')", example="男生")

class PredictionResponse(BaseModel):
    risk_probability: float
    is_diabetic: int
    risk_level: str
    clinical_advice: str

class TrainRequest(BaseModel):
    C: float = Field(default=1.0, gt=0, description="正則化強度逆數 C")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="測試集比例")


@app.post("/predict", response_model=PredictionResponse, summary="單筆病患風險預測")
def predict_endpoint(patient: PatientData):
    gender_num = 1 if patient.gender == "男生" else 0
    raw_features = np.array([[patient.age, patient.weight, patient.blood_sugar, gender_num]])
    
    scaled_features = scaler.transform(raw_features)
    prob = float(model.predict_proba(scaled_features)[0, 1])
    pred = int(prob >= 0.5)
    
    if prob >= 0.7:
        risk_level = "🔴 極高風險 (High Risk)"
        advice = "空腹血糖與生化數據偏高，強烈建議盡速安排糖化血色素 (HbA1c) 檢測與新陳代謝科門診複診。"
    elif prob >= 0.4:
        risk_level = "🟡 中度風險 (Moderate Risk)"
        advice = "處於糖尿病前期或潛在風險群，建議注意飲食熱量控制並進行生活型態衛教。"
    else:
        risk_level = "🟢 低風險 (Low Risk)"
        advice = "生理指標目前落於良好範圍，請持續維持規律運動與健康飲食。"
        
    return PredictionResponse(
        risk_probability=round(prob, 4),
        is_diabetic=pred,
        risk_level=risk_level,
        clinical_advice=advice
    )


@app.post("/train", summary="線上重訓練模型")
def train_endpoint(req: TrainRequest):
    global model, scaler, odds_ratios, meta
    new_meta = train_and_save_model(
        test_size=req.test_size,
        C=req.C,
        model_path=MODEL_PATH
    )
    model = new_meta["model"]
    scaler = new_meta["scaler"]
    odds_ratios = new_meta["odds_ratios"]
    meta = new_meta
    return {
        "status": "success",
        "message": "模型線上重新訓練完成並已熱載入！",
        "train_accuracy": new_meta["train_accuracy"],
        "test_accuracy": new_meta["test_accuracy"],
        "timestamp": new_meta["timestamp"]
    }


# 4. Gradio 互動介面
def gradio_predict(age, weight, blood_sugar, gender):
    gender_num = 1 if gender == "男生" else 0
    raw_features = np.array([[age, weight, blood_sugar, gender_num]])
    scaled_features = scaler.transform(raw_features)
    prob = float(model.predict_proba(scaled_features)[0, 1])
    
    prob_percent = f"{prob * 100:.1f}%"
    if prob >= 0.7:
        badge = "🔴 【極高風險】"
        advice = "臨床建議：空腹血糖值或年齡權重偏高，建議安排進一步 OGTT 耐糖試驗與新陳代謝專科追蹤。"
    elif prob >= 0.4:
        badge = "🟡 【中度警示】"
        advice = "臨床建議：數值接近前期臨界值，建議每半年複驗空腹血糖並落實低 GI 飲食。"
    else:
        badge = "🟢 【低風險健康】"
        advice = "臨床建議：各項指標良好，持續維持均衡營養與定期健檢。"
        
    # 繪製勝算比圖
    fig, ax = plt.subplots(figsize=(6, 3))
    features = list(odds_ratios.keys())
    ors = list(odds_ratios.values())
    bars = ax.barh(features, ors, color="teal")
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="基準 (OR=1.0)")
    ax.set_title("臨床危險因子勝算比 (Odds Ratio)", fontproperties=font_prop, fontsize=11)
    ax.set_xlabel("勝算比 (OR)", fontproperties=font_prop, fontsize=10)
    ax.legend(prop=font_prop, fontsize=9)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    
    return prob_percent, badge, advice, fig


with gr.Blocks(title="糖尿病患病風險評估系統", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 智慧醫療：糖尿病患病風險即時評估系統")
    gr.Markdown("本系統採用臨床可解釋性極高的 **邏輯迴歸（Logistic Regression）** 模型，提供即時患病風險機率預估與危險因子勝算比（Odds Ratio）剖析。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 病患臨床指標輸入")
            in_age = gr.Slider(minimum=10, maximum=100, value=45, step=1, label="年齡 (歲)")
            in_weight = gr.Slider(minimum=30, maximum=160, value=75, step=0.5, label="體重 (公斤)")
            in_sugar = gr.Slider(minimum=50, maximum=250, value=115, step=1, label="空腹血糖 (mg/dL)")
            in_gender = gr.Radio(choices=["男生", "女生"], value="男生", label="生理性別")
            btn_predict = gr.Button("開始臨床評估", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 評估診斷與分析")
            out_prob = gr.Label(label="預估患病機率")
            out_badge = gr.Textbox(label="風險等級", interactive=False)
            out_advice = gr.Textbox(label="臨床處置建議", interactive=False, lines=3)
            out_plot = gr.Plot(label="危險因子勝算比分析")
            
    btn_predict.click(
        fn=gradio_predict,
        inputs=[in_age, in_weight, in_sugar, in_gender],
        outputs=[out_prob, out_badge, out_advice, out_plot]
    )

# 掛載 Gradio 到 FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
