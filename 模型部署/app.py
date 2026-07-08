import os
import sys

# 將當前檔案所在目錄加入 sys.path，確保不論在本地或雲端從哪裡啟動，相對導入都能正常運作
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import joblib
from fastapi import FastAPI, HTTPException
import gradio as gr
from pydantic import BaseModel, Field

# ==========================================
# 1. 載入訓練好的模型
# ==========================================
model_path = os.path.join(os.path.dirname(__file__), "iris_model.joblib")
if not os.path.exists(model_path):
    print("未檢測到模型檔案，正在自動執行訓練以生成 iris_model.joblib...")
    try:
        from train_save import train_and_save_model
        train_and_save_model()
    except Exception as e:
        raise RuntimeError(f"自動訓練模型失敗: {str(e)}")

# 載入模型與類別標籤
model_data = joblib.load(model_path)
model = model_data["model"]
target_names = model_data["target_names"]
print("模型與類別標籤成功載入！")


# ==========================================
# 2. 建立 FastAPI 應用與 Pydantic 格式定義
# ==========================================
app = FastAPI(
    title="Iris 鳶尾花預測服務 API",
    description="這是一個結合 FastAPI 與 Gradio 的機器學習部署範例，同時提供 Web UI 與 API 端點。",
    version="1.0.0",
)


# 定義 API 輸入格式
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


# 定義 API 輸出格式
class IrisOutput(BaseModel):
    prediction_id: int = Field(..., description="預測類別 ID")
    prediction_label: str = Field(..., description="預測類別名稱")
    probabilities: dict[str, float] = Field(..., description="各類別預測機率")


# FastAPI 預測 API 端點
@app.post("/predict", response_model=IrisOutput)
def predict_api(payload: IrisInput):
    """
    提供給外部程式調用的預測 API 端點。
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
        # 預測類別
        pred_id = int(model.predict(features)[0])
        pred_label = target_names[pred_id]

        # 預測機率
        probs = model.predict_proba(features)[0]
        prob_dict = {target_names[i]: float(p) for i, p in enumerate(probs)}

        return IrisOutput(
            prediction_id=pred_id,
            prediction_label=pred_label,
            probabilities=prob_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")


# ==========================================
# 3. 建立 Gradio 網頁介面 (Web UI)
# ==========================================
def predict_gui(sepal_len, sepal_wid, petal_len, petal_wid):
    """
    Gradio 網頁介面專用的預測處理函數。
    """
    features = [[sepal_len, sepal_wid, petal_len, petal_wid]]

    # 進行預測
    pred_id = int(model.predict(features)[0])
    pred_label = target_names[pred_id]

    # 取得預測機率
    probs = model.predict_proba(features)[0]

    # 格式化輸出結果
    result_text = f"### 預測結果: **{pred_label.upper()}** (類別 ID: {pred_id})\n\n"
    result_text += "**各類別預測機率：**\n"
    for i, prob in enumerate(probs):
        result_text += f"- {target_names[i]}: {prob*100:.2f}%\n"

    return result_text


# 設計 Gradio 介面外觀與輸入組件
demo = gr.Interface(
    fn=predict_gui,
    inputs=[
        gr.Slider(minimum=0.1, maximum=10.0, value=5.1, label="花萼長度 Sepal Length (cm)"),
        gr.Slider(minimum=0.1, maximum=10.0, value=3.5, label="花萼寬度 Sepal Width (cm)"),
        gr.Slider(minimum=0.1, maximum=10.0, value=1.4, label="花瓣長度 Petal Length (cm)"),
        gr.Slider(minimum=0.1, maximum=10.0, value=0.2, label="花瓣寬度 Petal Width (cm)"),
    ],
    outputs=gr.Markdown(label="預測結果"),
    title="🌸 Iris 鳶尾花即時預測系統",
    description="調整左側的花卉特徵數值，系統將即時透過隨機森林模型預測其鳶尾花品種。",
    examples=[
        [5.1, 3.5, 1.4, 0.2],  # setosa 典型數值
        [6.0, 3.0, 4.8, 1.8],  # versicolor 典型數值
        [6.9, 3.1, 5.4, 2.1],  # virginica 典型數值
    ],
)

# ==========================================
# 4. 將 Gradio UI 掛載到 FastAPI 上
# ==========================================
# 掛載後，造訪 "/" 會開啟 Gradio 網頁，而外部呼叫 "/predict" 依然可以使用 FastAPI API
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    # 本地啟動指令：python app.py
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
