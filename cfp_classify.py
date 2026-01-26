import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import os
import logging

# 設定 Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
IMG_SIZE = 384
MODEL_CROP_TOL = 7

#  3 類標籤 (順序須與訓練時一致)
LABELS_LIST = ['y_dr', 'y_glaucoma', 'y_cataract']

# 定義模型架構 (與訓練時一致)(Concat CLS + GAP)
class ViTMultiLabel(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        # 輸入特徵為 2 倍 (cls_token & gap_feat)
        self.head = nn.Linear(backbone.num_features * 2, num_labels)
        
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        # feats shape: [batch, patches+1, embed_dim]
        cls_token = feats[:, 0]
        gap_feat = feats[:, 1:].mean(dim=1)
        combined_feat = torch.cat([cls_token, gap_feat], dim=1)
        return self.head(combined_feat)

# 前處理定義
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def crop_image_from_gray(img_arr, tol=7):
    """
    OpenCV 自動去黑邊邏輯
    回傳 (cropped_img, crop_coords)
    crop_coords 格式: (y_min, y_max, x_min, x_max) 或 None (若沒裁切)
    """
    if img_arr.ndim == 2:
        mask = img_arr > tol
        return img_arr[np.ix_(mask.any(1), mask.any(0))], None # 灰階暫不處理複雜回傳
    elif img_arr.ndim == 3:
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img_arr[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img_arr, None # 沒切
        else:
            img1 = img_arr[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img_arr[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img_arr[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
            
            # 計算裁切範圍
            rows = mask.any(1)
            cols = mask.any(0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            # y_max 和 x_max 需要 +1 才能包含最後一個像素
            return img, (y_min, y_max + 1, x_min, x_max + 1)
            
    return img, None

def load_cfp_model(model_path: str, device: torch.device):
    """
    載入模型權重，固定 num_classes=3 (由 Main Lifespan 呼叫)
    """
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading RETFound model from {model_path} ...")
    
    num_classes = len(LABELS_LIST)
    
    # 建立骨幹
    backbone = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0, img_size=IMG_SIZE)
    model = ViTMultiLabel(backbone, num_classes)
    
    # 載入權重
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    # 相容性處理：檢查是否有 'model' key
    state_dict = checkpoint.get('model', checkpoint)
    # 寬鬆載入以避免細微層名差異
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"⚠️ Missing keys: {missing}")
    
    model.to(device)
    if device.type == 'cuda':
        model.half() # 使用 FP16
    
    model.eval()
    logger.info("✅ Model loaded successfully.")
    return model

def predict_cfp(model, image_bytes: bytes, device: torch.device):
    """
    執行預測 (包含 TTA)
    回傳: (probs_numpy, processed_tensor, rgb_img_float, original_size)
    """
    # 1. 解碼圖片
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 紀錄最原始尺寸 (Full Size)
    full_h, full_w = img.shape[:2]
    
    # 2. Preprocess (Crop + Resize)
    cropped_img, crop_coords = crop_image_from_gray(img, tol=MODEL_CROP_TOL)
    pil_img = Image.fromarray(cropped_img)
    pil_img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC)
    
    # 用於 XAI 的原始影像 (Float, 0-1)
    img_display = np.array(pil_img_resized)
    rgb_img_float = np.float32(img_display) / 255.0
    
    # 3. 轉 Tensor (TTA: Original + Flip)
    input_tensor = val_transform(pil_img).unsqueeze(0).to(device)
    # 建立翻轉圖 tensor
    pil_img_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    input_tensor_flip = val_transform(pil_img_flip).unsqueeze(0).to(device)
    # 半精度處理
    if next(model.parameters()).dtype == torch.float16:
        input_tensor = input_tensor.half()
        input_tensor_flip = input_tensor_flip.half() 
        
    # 4. 推論
    with torch.no_grad():
        logits1 = model(input_tensor)
        probs1 = torch.sigmoid(logits1)
        
        logits2 = model(input_tensor_flip)
        probs2 = torch.sigmoid(logits2)
        
        # Max-TTA
        probs_tensor = torch.max(probs1, probs2)
        probs = probs_tensor[0].float().cpu().numpy()
        

    # 打包還原資訊 (沒裁切視為全圖)
    if crop_coords is None:
        crop_coords = (0, full_h, 0, full_w)
        
    restore_info = {
        "full_shape": (full_w, full_h), # PIL/OpenCV (Width, Height)
        "crop_coords": crop_coords      # (y1, y2, x1, x2)
    }
        
    # 回傳 restore_info、rgb_img_float
    rgb_img_float = np.float32(np.array(pil_img_resized)) / 255.0
    
    return probs, input_tensor, rgb_img_float, restore_info