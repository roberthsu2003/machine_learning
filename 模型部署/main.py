from contextlib import asynccontextmanager
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 定義 Lifespan 管理模型的載入
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時：載入模型
    model_path = os.path.join(os.path.dirname(__file__), "iris_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型檔案: {model_path}，請先運行 train_save.py 生成模型。"
        )

    try:
        model_data = joblib.load(model_path)
        app.state.model = model_data["model"]
        app.state.target_names = model_data["target_names"]
        print("模型與類別標籤成功載入！")
    except Exception as e:
        raise RuntimeError(f"載入模型時發生錯誤: {str(e)}")

    yield
    # 關閉時：可在此處釋放資源 (如果有需要的話)
    print("服務正在關閉...")


# 建立 FastAPI 實例
app = FastAPI(
    title="Iris 鳶尾花預測服務 API",
    description="這是一個使用 Scikit-Learn 訓練、並透過 FastAPI 部署的機器學習預測服務。",
    version="1.0.0",
    lifespan=lifespan,
)


# 定義輸入格式 (Pydantic Schema)
class IrisInput(BaseModel):
    sepal_length: float = Field(
        ..., description="花萼長度 (Sepal Length in cm)", ge=0.1, le=10.0
    )
    sepal_width: float = Field(
        ..., description="花萼寬度 (Sepal Width in cm)", ge=0.1, le=10.0
    )
    petal_length: float = Field(
        ..., description="花瓣長度 (Petal Length in cm)", ge=0.1, le=10.0
    )
    petal_width: float = Field(
        ..., description="花瓣寬度 (Petal Width in cm)", ge=0.1, le=10.0
    )

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


# 定義輸出格式 (Pydantic Schema)
class IrisOutput(BaseModel):
    prediction_id: int = Field(..., description="預測類別 ID (0, 1, 2)")
    prediction_label: str = Field(
        ..., description="預測類別標籤 ('setosa', 'versicolor', 'virginica')"
    )
    probabilities: dict[str, float] = Field(
        ..., description="各類別預測機率分布"
    )


# 根目錄歡迎與健康檢查端點
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "歡迎來到 Iris 鳶尾花預測 API！請造訪 /docs 查看 API 文件並進行測試。",
    }


# 預測端點
@app.post("/predict", response_model=IrisOutput)
def predict(payload: IrisInput):
    # 從 app.state 取得已載入的模型與標籤
    model = app.state.model
    target_names = app.state.target_names

    # 將輸入轉換為模型所需的特徵二維陣列
    features = [
        [
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]
    ]

    try:
        # 進行預測
        prediction_id = int(model.predict(features)[0])
        prediction_label = target_names[prediction_id]

        # 進行機率預測
        probs = model.predict_proba(features)[0]
        probabilities = {
            target_names[i]: float(prob) for i, prob in enumerate(probs)
        }

        return IrisOutput(
            prediction_id=prediction_id,
            prediction_label=prediction_label,
            probabilities=probabilities,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"模型預測過程中發生錯誤: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # 本地直接執行時啟動 Uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
