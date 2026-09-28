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

# 1. 載入模型
MODEL_PATH = "heart_failure_model.joblib"
if not os.path.exists(MODEL_PATH):
    train_and_save_model(model_path=MODEL_PATH)

meta = joblib.load(MODEL_PATH)
model = meta["model"]
feature_importances = meta["feature_importances"]

# 中文字型設定
font_path = "ChineseFont.ttf" if os.path.exists("ChineseFont.ttf") else "../../source_data/ChineseFont.ttf"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
    plt.rcParams["axes.unicode_minus"] = False
else:
    font_prop = None

# 2. FastAPI
app = FastAPI(
    title="心臟衰竭病患生存風險預測 API (Random Forest)",
    description="利用隨機森林進行心血管重症風險分級與臨床特徵重要性解析之微服務",
    version="1.0.0"
)

class HeartPatientData(BaseModel):
    age: float = Field(..., ge=18, le=110, description="年齡", example=65)
    anaemia: int = Field(..., ge=0, le=1, description="貧血 (0:無, 1:有)", example=0)
    cpk: float = Field(..., ge=10, le=10000, description="肌酸激酶 CPK (mcg/L)", example=160)
    diabetes: int = Field(..., ge=0, le=1, description="糖尿病史 (0:無, 1:有)", example=0)
    ejection_fraction: float = Field(..., ge=10, le=80, description="射血分數 EF (%)", example=35)
    high_blood_pressure: int = Field(..., ge=0, le=1, description="高血壓史 (0:無, 1:有)", example=1)
    platelets: float = Field(..., ge=20000, le=900000, description="血小板 (platelets/mL)", example=250000)
    serum_creatinine: float = Field(..., ge=0.4, le=10.0, description="血清肌酸酐 (mg/dL)", example=1.8)
    serum_sodium: float = Field(..., ge=110, le=150, description="血清鈉 (mEq/L)", example=136)
    sex: int = Field(..., ge=0, le=1, description="生理性別 (0:女性, 1:男性)", example=1)
    smoking: int = Field(..., ge=0, le=1, description="抽菸 (0:無, 1:有)", example=0)

class RiskPredictionResponse(BaseModel):
    mortality_risk_probability: float
    risk_category: str
    clinical_alert: str

class TrainRequest(BaseModel):
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_depth: int = Field(default=4, ge=1, le=20)


@app.post("/predict", response_model=RiskPredictionResponse, summary="單筆病患重症風險預測")
def predict_endpoint(patient: HeartPatientData):
    raw_features = np.array([[
        patient.age, patient.anaemia, patient.cpk, patient.diabetes,
        patient.ejection_fraction, patient.high_blood_pressure, patient.platelets,
        patient.serum_creatinine, patient.serum_sodium, patient.sex, patient.smoking
    ]])
    
    prob = float(model.predict_proba(raw_features)[0, 1])
    
    if prob >= 0.6:
        category = "🔴 重症高死亡風險 (Critical High Risk)"
        alert = "肌酸酐或射血分數呈現心腎衰竭徵兆，建議即刻送加護病房 (ICU) 或安排急診心導管/強心利尿劑處置。"
    elif prob >= 0.35:
        category = "🟡 中度監控風險 (Moderate Risk)"
        alert = "心臟收縮功能受損，需密切追蹤電解質平衡與每日體重尿量變化。"
    else:
        category = "🟢 病情相對平穩 (Stable / Low Risk)"
        alert = "各項器官灌流指數尚可，持續維持常規慢性心衰竭藥物治療。"
        
    return RiskPredictionResponse(
        mortality_risk_probability=round(prob, 4),
        risk_category=category,
        clinical_alert=alert
    )


@app.post("/train", summary="線上熱調參重訓模型")
def train_endpoint(req: TrainRequest):
    global model, feature_importances, meta
    new_meta = train_and_save_model(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        model_path=MODEL_PATH
    )
    model = new_meta["model"]
    feature_importances = new_meta["feature_importances"]
    meta = new_meta
    return {
        "status": "success",
        "message": f"隨機森林模型重訓完成 (樹數量: {req.n_estimators}, 最大深度: {req.max_depth})",
        "train_accuracy": new_meta["train_accuracy"],
        "test_accuracy": new_meta["test_accuracy"],
        "timestamp": new_meta["timestamp"]
    }


# 3. Gradio
def gradio_predict(age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking):
    anaemia_num = 1 if anaemia == "有" else 0
    diabetes_num = 1 if diabetes == "有" else 0
    hbp_num = 1 if hbp == "有" else 0
    sex_num = 1 if sex == "男性" else 0
    smoking_num = 1 if smoking == "有" else 0
    
    raw = np.array([[
        age, anaemia_num, cpk, diabetes_num, ef, hbp_num, platelets,
        creatinine, sodium, sex_num, smoking_num
    ]])
    
    prob = float(model.predict_proba(raw)[0, 1])
    prob_str = f"{prob * 100:.1f}%"
    
    if prob >= 0.6:
        status = "🔴 【重症高死亡風險】"
        rec = "⚠️ 警示：血清肌酸酐或心臟射血分數落入危險區間，強烈建議啟動加護病房監護與心臟重症專科會診。"
    elif prob >= 0.35:
        status = "🟡 【中度監控風險】"
        rec = "⚠️ 提示：病患收縮功能或代謝指數有衰退跡象，建議 48 小時內複驗肌酸酐與射血分數。"
    else:
        status = "🟢 【平穩低風險】"
        rec = "✅ 提示：器官灌流與心臟泵血指標相對平穩，常規門診追蹤即可。"
        
    # 繪製特徵重要性圖
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_items]
    scores = [x[1] for x in sorted_items]
    bars = ax.barh(names, scores, color="mediumpurple")
    ax.set_title("心臟衰竭隨機森林特徵重要性權重", fontproperties=font_prop, fontsize=11)
    ax.set_xlabel("重要性權重", fontproperties=font_prop, fontsize=10)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    
    return prob_str, status, rec, fig


with gr.Blocks(title="心臟衰竭重症風險預測系統", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🫀 智慧醫療：心臟衰竭病患重症風險分級系統")
    gr.Markdown("本系統採用 100 棵樹的 **隨機森林（Random Forest）** 分類器，評估急診與住院病患之惡化風險，並即時解析關鍵檢驗權重。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🩺 臨床急診生理檢驗輸入")
            in_age = gr.Slider(20, 95, value=65, step=1, label="年齡 (歲)")
            in_ef = gr.Slider(10, 75, value=30, step=1, label="心臟射血分數 EF (%) [越低越危險]")
            in_creatinine = gr.Slider(0.5, 9.0, value=1.8, step=0.1, label="血清肌酸酐 (mg/dL) [越高越危險]")
            in_cpk = gr.Slider(20, 5000, value=250, step=10, label="肌酸激酶 CPK (mcg/L)")
            in_sodium = gr.Slider(115, 148, value=135, step=1, label="血清鈉離子 (mEq/L)")
            in_platelets = gr.Slider(50000, 600000, value=250000, step=10000, label="血小板數量")
            
            with gr.Row():
                in_sex = gr.Radio(["男性", "女性"], value="男性", label="性別")
                in_hbp = gr.Radio(["無", "有"], value="有", label="高血壓史")
            with gr.Row():
                in_diabetes = gr.Radio(["無", "有"], value="無", label="糖尿病史")
                in_anaemia = gr.Radio(["無", "有"], value="無", label="貧血")
                in_smoking = gr.Radio(["無", "有"], value="無", label="抽菸習慣")
                
            btn_run = gr.Button("開始重症風險推論", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 風險分級與特徵影響力")
            out_p = gr.Label(label="預估惡化/死亡風險機率")
            out_s = gr.Textbox(label="重症風險分級", interactive=False)
            out_r = gr.Textbox(label="急診臨床處置指引", interactive=False, lines=3)
            out_f = gr.Plot(label="臨床指標重要性排行")
            
    btn_run.click(
        fn=gradio_predict,
        inputs=[in_age, in_anaemia, in_cpk, in_diabetes, in_ef, in_hbp, in_platelets, in_creatinine, in_sodium, in_sex, in_smoking],
        outputs=[out_p, out_s, out_r, out_f]
    )

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
