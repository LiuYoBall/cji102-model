import uuid
import logging
from google.cloud import storage

# 引用您的模組 (確認 predict_cfp 內部是用 PIL 處理 bytes)
from cfp_classify import predict_cfp
from inference import analyze_results
from xai import generate_xai_image
# from segmentation import run_yolo_segmentation (預留)

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GCS 輸出設定 ---
# 這是用來存放「AI 分析結果圖片」的 Bucket
OUTPUT_BUCKET_NAME = "fundus-ai-project" 

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
    上傳圖片到 Output Bucket 並回傳公開 URL
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(OUTPUT_BUCKET_NAME) # 使用上方定義的變數
        blob = bucket.blob(destination_blob_name)
        
        # 上傳 (同步方法，但在 Cloud Run 並發量不高時可接受)
        blob.upload_from_string(file_data, content_type=content_type)
        
        logger.info(f"⬆️ Uploaded to: {destination_blob_name}")
        return blob.public_url
        
    except Exception as e:
        logger.error(f"❌ GCS Upload Error: {e}")
        return None

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
        # [注意] 確保 predict_cfp 內部使用 PIL.Image.open(io.BytesIO(file_bytes))
        probs, input_tensor, rgb_img_float = predict_cfp(model_cfp, file_bytes, device)
        
        # 3. 邏輯分析
        result_json = analyze_results(probs)
        
        # 4. XAI 熱力圖處理
        image_urls = {
            "original": gcs_path # 回傳原圖路徑給前端參考
        }
        
        target_idx = result_json.get('target_cam_idx')
        
        # 只有在需要時才生成 CAM
        if target_idx is not None:
            cam_bytes = generate_xai_image(model_cfp, input_tensor, target_idx, rgb_img_float)
            
            if cam_bytes:
                cam_url = await upload_to_gcs(cam_bytes, f"results/{request_id}_cam.jpg")
                if cam_url:
                    image_urls['cam_url'] = cam_url

        # 5. (Optional) YOLO 預留區
        if result_json.get('is_dr', False) and model_yolo:
             # yolo_bytes = run_yolo_segmentation(model_yolo, file_bytes) # 傳入 bytes 讓 yolo 自己處理
             # yolo_url = await upload_to_gcs(yolo_bytes, f"results/{request_id}_yolo.jpg")
             pass

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