from cfp_classify import predict_cfp
from inference import analyze_results
from xai import generate_xai_image
# from segmentation import run_yolo_segmentation (預留)
from google.cloud import storage
import uuid
import torch

# GCS 設定 (請依實際情況修改)
BUCKET_NAME = "fundus-ai-project" # TODO: 修改這裡

async def upload_to_gcs(file_data: bytes, destination_blob_name: str, content_type="image/jpeg"):
    """
    上傳圖片到 GCS 並回傳公開 URL (或 Signed URL)
    """
    # 如果是在本地測試，可以先回傳假 URL
    # return f"https://mock-storage/{destination_blob_name}"

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(file_data, content_type=content_type)
        return blob.public_url
    except Exception as e:
        print(f"GCS Upload Error: {e}")
        return None

async def process_fundus_image(
    file_bytes: bytes, 
    model_cfp, 
    model_yolo, 
    device: torch.device
):
    """
    主控流程：
    1. CFP 分類
    2. 邏輯判斷
    3. (若需要) YOLO 切割
    4. XAI 繪圖
    5. 上傳結果圖片至 GCS
    """
    
    # 1. CFP 分類
    probs, input_tensor, rgb_img_float = predict_cfp(model_cfp, file_bytes, device)
    
    # 2. 邏輯分析
    result_json = analyze_results(probs)
    
    # 生成唯一 ID
    request_id = str(uuid.uuid4())[:8]
    image_urls = {}
    
    # 3. 處理 XAI (Cam Map)
    target_idx = result_json['target_cam_idx']
    cam_bytes = generate_xai_image(model_cfp, input_tensor, target_idx, rgb_img_float)
    
    # 平行上傳 XAI 圖片
    if cam_bytes:
        cam_url = await upload_to_gcs(cam_bytes, f"results/{request_id}_cam.jpg")
        image_urls['cam_url'] = cam_url

    # 4. (Optional) 處理 YOLO - 只有當判定為 DR 時才執行
    if result_json['is_dr'] and model_yolo:
        # TODO: 這裡呼叫你的 segmentation.py
        # yolo_bytes = run_yolo_segmentation(model_yolo, rgb_img_float)
        # yolo_url = await upload_to_gcs(yolo_bytes, f"results/{request_id}_lesion.jpg")
        # image_urls['lesion_url'] = yolo_url
        pass

    # 合併結果
    final_response = {
        "request_id": request_id,
        "analysis": result_json,
        "images": image_urls
    }
    
    return final_response