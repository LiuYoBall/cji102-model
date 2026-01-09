import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# 引用我們寫好的模組
from cfp_classify import load_cfp_model
from controller import process_fundus_image

# --- 設定與全域變數 ---
# Cloud Run GCS Mount 路徑通常設為 /mnt/gcs_bucket_name
# 本地測試時，可改為你的本機路徑
MODEL_MOUNT_PATH = os.getenv("MODEL_MOUNT_PATH", "/mnt/models") 
CFP_MODEL_FILENAME = "0104_RETFound_inference.pth"
YOLO_MODEL_FILENAME = "best_yolo.pt" # 預留

# model本地路徑(測試用)
local_model_path = r"C:\Users\TMP-214\Desktop\deployment\model\0104_RETFound_inference.pth"

models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Lifespan: 啟動時載入模型 (關鍵優化) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting up... Device: {device}")
    
    # 建構完整路徑
    cfp_path = os.path.join(MODEL_MOUNT_PATH, CFP_MODEL_FILENAME)
    
    # 嘗試載入 CFP 模型
    try:
        if os.path.exists(cfp_path):
            # 優先嘗試 Cloud Run GCS 掛載路徑
            models["cfp"] = load_cfp_model(cfp_path, device)
            print(f"✅ CFP Model loaded from GCS Mount: {cfp_path}")
        elif os.path.exists(local_model_path):
            # 其次嘗試本地絕對路徑 (Local Test)
            print(f"⚠️ GCS Mount not found. Loading from Local Path: {local_model_path}")
            models["cfp"] = load_cfp_model(local_model_path, device)
        else:
            # 兩者都找不到 (避免程式崩潰，但標記服務不可用)
            print("❌ Critical Error: No model file found in GCS or Local path.")
            models["cfp"] = None
    except Exception as e:
        print(f"❌ Error loading CFP model: {e}")
        models["cfp"] = None

    # (預留) 載入模型
    # models["yolo"] = load_yolo_model(...)
    models["yolo"] = None
    models["oct"] = None  

    yield
    
    # 關閉時清理
    print("🛑 Shutting down. Clearing GPU memory...")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# --- 2. Health Check (Cloud Run 需要) ---
@app.get("/")
def health_check():
    status = "ready" if models.get("cfp") is not None else "model_missing"
    return {"status": status, "device": str(device)}

# --- 3. cfp API 入口 ---
@app.post("/predict/cfp")
async def predict_cfp_endpoint(
    file: UploadFile = File(...),
    # background_tasks: BackgroundTasks # 若需背景上傳可啟用
):
    # 1. 檢查模型是否就緒
    if models.get("cfp") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 2. 驗證檔案格式
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG supported.")

    try:
        # 3. 讀取檔案內容
        file_bytes = await file.read()
        
        # 4. 呼叫 Controller 進行處理
        # 注意：process_fundus_image 是 async 的
        result = await process_fundus_image(
            file_bytes=file_bytes,
            model_cfp=models["cfp"],
            model_yolo=models["yolo"],
            device=device
        )
        
        return JSONResponse(content=result)

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# # --- 3. oct API 入口 ---
# @app.post("/predict/oct")

if __name__ == "__main__":
    import uvicorn
    # 本地測試啟動指令
    uvicorn.run(app, host="0.0.0.0", port=8080)