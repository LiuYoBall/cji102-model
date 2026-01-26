import cv2
import numpy as np
import logging
from ultralytics import YOLO
import torch

# --- 設定 Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= 1. 參數設定 =================
IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# 閾值
CLASS_CONF_THRESHOLDS = {
    0: 0.4,  # Hemorrhages
    1: 0.4,  # Hard Exudates
    2: 0.3,  # Soft Exudates
}

# 邏輯參數 (1024x1024)
MERGE_DIST_THRESH = 50   
ENABLE_SPLIT_LOGIC = True
MAX_AREA_RATIO = 0.1     
SPLIT_DIST_THRESH = 30    

# 視覺化參數 (BGR)
COLOR_MAP = {
    0: (255, 0, 0),    
    1: (255, 255, 0),    
    2: (0, 255, 255),  
}

# [動態調整] 根據原圖大小自動計算，此處僅為預設值
BASE_THICKNESS = 2 
BASE_FONT_SCALE = 0.5

CLASS_ABBR = {0: "Hem", 1: "HE", 2: "SE"}

# ================= 2. 邏輯運算函式 =================

def filter_boxes_by_class_conf(boxes):
    """依照類別閾值過濾框"""
    if len(boxes) == 0: return []
    if isinstance(boxes, torch.Tensor): boxes = boxes.cpu().numpy()
    
    valid_boxes = []
    for box in boxes:
        cls_id = int(box[5])
        conf = box[4]
        thresh = CLASS_CONF_THRESHOLDS.get(cls_id, DEFAULT_CONF)
        if conf >= thresh:
            valid_boxes.append(box)
    return np.array(valid_boxes)

def merge_nearby_boxes_by_class(boxes, dist_thresh=50):
    """通用合併函數"""
    if len(boxes) == 0: return []
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            cur_box = boxes[i].copy()
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                other_box = boxes[j]
                if cur_box[5] != other_box[5]: continue 
                
                cx1, cy1 = (cur_box[0]+cur_box[2])/2, (cur_box[1]+cur_box[3])/2
                cx2, cy2 = (other_box[0]+other_box[2])/2, (other_box[1]+other_box[3])/2
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                
                xx1 = max(cur_box[0], other_box[0])
                yy1 = max(cur_box[1], other_box[1])
                xx2 = min(cur_box[2], other_box[2])
                yy2 = min(cur_box[3], other_box[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter_area = w * h
                
                if inter_area > 0 or dist < dist_thresh:
                    cur_box[0] = min(cur_box[0], other_box[0])
                    cur_box[1] = min(cur_box[1], other_box[1])
                    cur_box[2] = max(cur_box[2], other_box[2])
                    cur_box[3] = max(cur_box[3], other_box[3])
                    cur_box[4] = max(cur_box[4], other_box[4])
                    used[j] = True
                    merged = True
            new_boxes.append(cur_box)
        boxes = new_boxes
    return np.array(boxes)

def process_oversized_boxes(merged_boxes, raw_boxes, img_w, img_h):
    """大框拆解邏輯"""
    if not ENABLE_SPLIT_LOGIC or len(merged_boxes) == 0:
        return merged_boxes

    final_output = []
    img_area = img_w * img_h
    
    for box in merged_boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        
        if area > (img_area * MAX_AREA_RATIO):
            internal_raws = []
            box_cls = int(box[5])
            
            for raw in raw_boxes:
                raw_cls = int(raw[5])
                if raw_cls != box_cls: continue 
                
                rx, ry = (raw[0]+raw[2])/2, (raw[1]+raw[3])/2
                if box[0] <= rx <= box[2] and box[1] <= ry <= box[3]:
                    internal_raws.append(raw)
            
            if len(internal_raws) > 0:
                split_results = merge_nearby_boxes_by_class(internal_raws, dist_thresh=SPLIT_DIST_THRESH)
                final_output.extend(split_results)
            else:
                final_output.append(box)
        else:
            final_output.append(box)
            
    return np.array(final_output)

# ================= 3. 圖像處理與座標轉換 =================

def preprocess_image_training_style(img, target_size=1024):
    """前處理：CLAHE -> Stretch Resize"""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        img_resized = cv2.resize(
            img_enhanced,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR
        )
        return img_resized
    except Exception as e:
        logger.warning(f"Preprocessing failed: {e}, using original resize.")
        return cv2.resize(img, (target_size, target_size))

def rescale_boxes(boxes, current_shape, original_shape):
    """
    將座標從 Model Size (1024x1024) 轉換回 原圖 Size
    current_shape: (1024, 1024)
    original_shape: (Height, Width)
    """
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = original_shape
    curr_h, curr_w = current_shape

    # 計算縮放比例
    scale_x = orig_w / curr_w
    scale_y = orig_h / curr_h

    rescaled_boxes = []
    for box in boxes:
        box = list(box) # 轉 list 方便修改
        # x1, x2 乘上 x比例
        box[0] *= scale_x
        box[2] *= scale_x
        # y1, y2 乘上 y比例
        box[1] *= scale_y
        box[3] *= scale_y
        rescaled_boxes.append(box)

    return np.array(rescaled_boxes)

def draw_boxes_on_original(img, boxes):
    """
    繪製邏輯：根據原圖大小自動調整字體與線條粗細
    """
    h, w = img.shape[:2]
    
    # 動態計算視覺參數
    # 基準是以 1024 為 2px，圖變大線條跟著變粗
    thickness = max(2, int(w / 500)) 
    font_scale = max(0.5, w / 2000)
    text_thickness = max(1, int(w / 1000))

    # 1. 繪製 Box 與 Score
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        cls_id = int(box[5])
        color = COLOR_MAP.get(cls_id, (255, 255, 255))
        
        # 畫空心框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)
        
        # 畫分數標籤
        score_text = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        
        # 確保文字背景框不會畫到圖片外面
        y_text_bg = max(0, y1 - th - 5)
        
        cv2.rectangle(img, (x1, y_text_bg), (x1 + tw, y_text_bg + th + 5), color, -1)
        cv2.putText(img, score_text, (x1, y_text_bg + th), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness)

    # 2. 繪製圖例 (Legend) - 左上角
    # 圖例大小也隨圖片縮放
    box_size = int(30 * (w / 1000))
    if box_size < 20: box_size = 20
    
    legend_x, legend_y = int(20 * (w/1000)), int(30 * (w/1000))
    text_gap = 10
    line_height = int(box_size * 1.5)
    
    # 背景框
    overlay = img.copy()
    bg_w = int(180 * (w/1000))
    if bg_w < 150: bg_w = 150
    bg_h = 20 + len(CLASS_ABBR) * line_height
    
    cv2.rectangle(overlay, (10, 10), (10 + bg_w, 10 + bg_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    
    for cls_id, name in CLASS_ABBR.items():
        color = COLOR_MAP.get(cls_id, (255, 255, 255))
        
        cv2.rectangle(img, (legend_x, legend_y), (legend_x + box_size, legend_y + box_size), color, -1)
        
        cv2.putText(img, name, (legend_x + box_size + text_gap, legend_y + box_size - int(box_size*0.1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), text_thickness)
        
        legend_y += line_height

    return img

# ================= 4. 主要接口 =================

def load_yolo_model(model_path: str, device: torch.device):
    """載入 YOLO 模型"""
    try:
        logger.info(f"Loading YOLO model from {model_path} ...")
        model = YOLO(model_path)
        model.to(device)
        logger.info("✅ YOLO model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise e

def run_yolo_segmentation(model, image_bytes: bytes):
    """
    流程：
    1. Decode 原圖 (保留)
    2. Preprocess -> 1024x1024 (給模型用)
    3. Inference & Logic -> 產生 1024 空間的 Boxes
    4. Rescale -> 將 Boxes 轉回原圖座標
    5. Draw -> 畫在原圖上
    """
    try:
        # 1. Decode 原圖
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 原解析度圖
        if img_original is None:
            raise ValueError("Failed to decode image bytes")
        
        # 取得原圖尺寸
        h_orig, w_orig = img_original.shape[:2]

        # 2. Preprocess (to 1024)
        img_for_ai = preprocess_image_training_style(img_original, target_size=IMG_SIZE)

        # 3. Inference (1024)
        results = model.predict(
            source=img_for_ai,
            imgsz=IMG_SIZE,
            conf=min(CLASS_CONF_THRESHOLDS.values()),
            verbose=False,
            save=False
        )
        
        # 4. 邏輯處理 (1024)
        raw_boxes = results[0].boxes.data
        filtered_boxes = filter_boxes_by_class_conf(raw_boxes)
        merged_boxes = merge_nearby_boxes_by_class(filtered_boxes, dist_thresh=MERGE_DIST_THRESH)
        final_boxes_1024 = process_oversized_boxes(merged_boxes, filtered_boxes, IMG_SIZE, IMG_SIZE)
        
        # 5. 座標回推 (1024 -> Original Size)
        final_boxes_orig = rescale_boxes(
            final_boxes_1024, 
            current_shape=(IMG_SIZE, IMG_SIZE), 
            original_shape=(h_orig, w_orig)
        )

        # 6. 繪圖 (img_original)
        final_img = draw_boxes_on_original(img_original, final_boxes_orig)

        # 7. Encode
        is_success, buffer = cv2.imencode(".jpg", final_img)
        if is_success:
            return buffer.tobytes()
        else:
            return None

    except Exception as e:
        logger.error(f"YOLO Processing Error: {e}")
        return None