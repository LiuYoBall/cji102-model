import asyncio
import functools
import uuid
import logging
import os
from google.cloud import storage

# 引用您的模組 (確認 predict_cfp 內部是用 PIL 處理 bytes)
from cfp_classify import predict_cfp
from inference import analyze_results
from xai import generate_xai_image
from segmentation import run_yolo_segmentation

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GCS 輸出設定 ---
OUTPUT_BUCKET_NAME = os.getenv("OUTPUT_BUCKET_NAME", "fundus-ai-project") 

def download_bytes_from_gcs(gcs_uri: str) -> bytes:
    """
    解析 GCS URI 並下載為原始 Bytes
    """
    try:
        storage_client = storage.Client()
        
        # 1. 解析路徑 (移除 gs:// 前綴)
        clean_uri = gcs_uri.replace("gs://", "")
        
        # 2. 分割 Bucket 與 Blob 名稱
        if "/" not in clean_uri:
            raise ValueError(f"Invalid GCS URI format: {gcs_uri}")    
        
        bucket_name, blob_name = clean_uri.split("/", 1)

        # 3. 下載
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        logger.info(f"⬇️ Downloading from Bucket: {bucket_name}, Blob: {blob_name}")
        return blob.download_as_bytes()
    
    except Exception as e:
        logger.error(f"❌ GCS Download Error: {e}")
        raise RuntimeError(f"Failed to download {gcs_uri}: {e}")

async def upload_to_gcs(file_data: bytes, destination_blob_name: str, content_type="image/jpeg"):
    """
    上傳圖片到 Output Bucket 並回傳 URL
    """
    def _sync_upload():
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(OUTPUT_BUCKET_NAME)
            blob = bucket.blob(destination_blob_name)
            
            # 這是同步操作，會花費時間
            blob.upload_from_string(file_data, content_type=content_type)
            
            logger.info(f"⬆️ Uploaded to: {destination_blob_name}")
            return blob.public_url
        except Exception as e:
            logger.error(f"❌ GCS Upload Error: {e}")
            return None

    # 取得當前的 Event Loop
    loop = asyncio.get_running_loop()
    
    # 關鍵：將同步函式丟到 ThreadPoolExecutor 執行
    # run_in_executor(None, ...) 的 None 代表使用預設的 ThreadPool
    result = await loop.run_in_executor(None, _sync_upload)
    
    return result

async def process_fundus_image(gcs_path: str, model_cfp, model_yolo, device):
    """
    主控流程 Controller (Bytes Version)
    """
    # 產生 Request ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing: {gcs_path}")

    try:
        # 1. 下載圖片 (Bytes)
        file_bytes = download_bytes_from_gcs(gcs_path)

        # 2. CFP 分類
        probs, input_tensor, rgb_img_float, restore_info = predict_cfp(model_cfp, file_bytes, device)
        
        # 3. 邏輯分析
        result_json = analyze_results(probs)
        
        # 4. XAI 熱力圖處理
        image_urls = {
            "original": gcs_path # 回傳原圖路徑給前端參考
        }
        
        target_idx = result_json.get('target_cam_idx')
        
        # 只有在需要時才生成 CAM
        if target_idx is not None:
            cam_bytes, raw_cam_bytes = generate_xai_image(model_cfp, input_tensor, target_idx, rgb_img_float, restore_info=restore_info)
            
            if cam_bytes:
                cam_url = await upload_to_gcs(cam_bytes, f"results/{request_id}_cam.jpg")
                if cam_url:
                    image_urls['cam_url'] = cam_url

            # 上傳 Resize 後的純熱力圖
            if raw_cam_bytes:
                raw_cam_url = await upload_to_gcs(raw_cam_bytes, f"results/{request_id}_cam_raw.jpg")
                if raw_cam_url:
                    image_urls['cam_raw_url'] = raw_cam_url

        # 5. YOLO 分割邏輯
        # 觸發條件：分析結果判斷為 DR (is_dr=True) 且 YOLO 模型已載入
        if result_json.get('is_dr', False) and model_yolo:
             logger.info(f"[{request_id}] DR detected, running YOLO segmentation...")
             try:
                 # 傳入 bytes 讓 yolo 模組處理
                 yolo_bytes = run_yolo_segmentation(model_yolo, file_bytes)
                 
                 # 上傳結果
                 if yolo_bytes:
                     yolo_url = await upload_to_gcs(yolo_bytes, f"results/{request_id}_yolo.jpg")
                     if yolo_url:
                         image_urls['yolo_url'] = yolo_url
             except Exception as e:
                 logger.error(f"[{request_id}] YOLO Error: {e}")
                 # YOLO 失敗不應影響主流程回傳，僅記錄錯誤即可

        # 6. 組裝最終回傳
        final_response = {
            "request_id": request_id,
            "status": "success",
            "analysis": result_json,
            "images": image_urls
        }
        
        return final_response

    except Exception as e:
        logger.error(f"[{request_id}] Critical Error: {e}")
        # 拋出錯誤讓 main.py 捕捉並回傳 500
        raise e