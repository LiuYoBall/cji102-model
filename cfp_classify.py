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

# 定義模型架構 (必須與訓練時一致)
class ViTMultiLabel(nn.Module):
    def __init__(self, bb, n_cls):
        super().__init__()
        self.backbone = bb
        self.head = nn.Linear(bb.num_features, n_cls)
        
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            feats = feats[:, 1:].mean(dim=1) # GAP
        return self.head(feats)

# 前處理定義
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def crop_image_from_gray(img_arr, tol=7):
    """OpenCV 自動去黑邊邏輯"""
    if img_arr.ndim == 2:
        mask = img_arr > tol
        return img_arr[np.ix_(mask.any(1), mask.any(0))]
    elif img_arr.ndim == 3:
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img_arr[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img_arr
        else:
            rows = mask.any(1)
            cols = mask.any(0)
            return img_arr[rows][:, cols]
    return img_arr

def load_cfp_model(model_path: str, device: torch.device, num_classes: int = 4):
    """
    載入模型權重 (由 Main Lifespan 呼叫)
    """
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading RETFound model from {model_path} ...")
    
    # 建立骨幹
    backbone = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0, img_size=IMG_SIZE)
    model = ViTMultiLabel(backbone, num_classes)
    
    # 載入權重
    checkpoint = torch.load(model_path, map_location="cpu")
    # 相容性處理：檢查是否有 'model' key
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    
    model.to(device)
    if device.type == 'cuda':
        model.half() # 使用 FP16
    
    model.eval()
    logger.info("✅ Model loaded successfully.")
    return model

def predict_cfp(model, image_bytes: bytes, device: torch.device):
    """
    執行預測 (包含 TTA)
    回傳: (probs_numpy, processed_tensor, original_rgb_float_image)
    """
    # 1. 解碼圖片
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. 前處理 (Crop + Resize)
    cropped_img = crop_image_from_gray(img, tol=MODEL_CROP_TOL)
    pil_img = Image.fromarray(cropped_img)
    pil_img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    
    # 用於 XAI 的原始影像 (Float, 0-1)
    img_display = np.array(pil_img_resized)
    rgb_img_float = np.float32(img_display) / 255.0
    
    # 3. 轉 Tensor (TTA: Original + Flip)
    input_tensor = val_transform(pil_img).unsqueeze(0).to(device)
    pil_img_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    input_tensor_flip = val_transform(pil_img_flip).unsqueeze(0).to(device)
    
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
        
    return probs, input_tensor, rgb_img_float