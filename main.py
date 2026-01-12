import os
import torch
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional

# 引用我們寫好的模組
# controller 已經修改為接收 gcs_path 
from cfp_classify import load_cfp_model
from controller import process_fundus_image 

# --- 設定 Log ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 設定路徑與變數 ---
MODEL_MOUNT_PATH = os.getenv("MODEL_MOUNT_PATH", "/mnt/models") 
CFP_MODEL_FILENAME = "0104_RETFound_inference.pth"
YOLO_MODEL_FILENAME = "best_yolo.pt" # 預留
OCT_MODEL_FILENAME = "oct_model.pth" # 預留

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Pydantic Model: 定義輸入 JSON 格式 ---
class PredictionRequest(BaseModel):
    image_gcs_path: str = Field(..., description="GCS上的圖片路徑")
    request_id: Optional[str] = None

# --- 1. Lifespan: 啟動與關閉生命週期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting up... Device: {device}")
    
    # 初始化模型容器
    models = {
        "cfp": None,
        "yolo": None,
        "oct": None
    }

    # 建構路徑
    cfp_path = os.path.join(MODEL_MOUNT_PATH, CFP_MODEL_FILENAME)
    # oct_path = os.path.join(MODEL_MOUNT_PATH, OCT_MODEL_FILENAME) # 預留

    # --- 載入 CFP 模型 ---
    try:
        if os.path.exists(cfp_path):
            models["cfp"] = load_cfp_model(cfp_path, device)
            logger.info(f"✅ CFP Model loaded from: {cfp_path}")
        else:
            logger.error(f"❌ Critical Error: CFP model not found at {cfp_path}")
            # 在生產環境中，這裡可以選擇是否要 raise error 阻止啟動
    except Exception as e:
        logger.error(f"❌ Error loading CFP model: {e}")

    # --- (預留) 載入 YOLO 模型 ---
    # try:
    #     models["yolo"] = load_yolo_model(yolo_path, device)
    # except...

    # --- (預留) 載入 OCT 模型 ---
    # try:
    #     if os.path.exists(oct_path):
    #          models["oct"] = load_oct_model(oct_path, device)
    # except...

    # [關鍵] 將模型存入 app.state，供全域存取
    app.state.models = models

    yield  # 應用程式運行中...

    # --- 關閉時清理 ---
    logger.info("🛑 Shutting down. Clearing GPU memory...")
    app.state.models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# --- 2. Health Check ---
@app.get("/")
def health_check(request: Request):
    """檢查服務與模型狀態"""
    models = request.app.state.models
    status = "ready" if models.get("cfp") is not None else "partial_service"
    
    return {
        "status": status, 
        "device": str(device),
        "loaded_models": [k for k, v in models.items() if v is not None]
    }

# --- 3. 預測入口 ---
# 分開路由，讓 API 文件(Swagger)更清晰

@app.post("/predict/cfp")
async def predict_cfp_endpoint(
    request: Request, 
    payload: PredictionRequest 
):
    models = request.app.state.models
    if models.get("cfp") is None:
        raise HTTPException(status_code=503, detail="CFP Model not initialized")

    try:
        # 呼叫 Controller
        # 注意：Controller 內部會負責產生 CAM 和 YOLO 圖，並回傳包含這些 URL 的 dict
        result = await process_fundus_image(
            gcs_path=payload.image_gcs_path,
            model_cfp=models["cfp"],
            model_yolo=models["yolo"], 
            device=device
        )
        
        # 回填 request_id
        if payload.request_id:
            result["request_id"] = payload.request_id
            
        return JSONResponse(content=result)

    except Exception as e:
        # Log error...
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/oct")
async def predict_oct_endpoint(
    request: Request, 
    payload: PredictionRequest
):
    return JSONResponse(content={
        "status": "pending",
        "message": "OCT inference not implemented",
        "source": payload.image_gcs_path
    })

if __name__ == "__main__":
    import uvicorn
    # 本地測試啟動指令
    uvicorn.run(app, host="0.0.0.0", port=8080)