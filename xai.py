import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch

def generate_xai_image(model, input_tensor, target_idx, rgb_img_float, restore_info=None):
    """
    生成 GradCAM 圖片
    回傳: (overlay_bytes, raw_heatmap_bytes)
    """
    # 針對ViTLarge，Layer Norm 通常在 blocks 最後
    target_layers = [model.backbone.blocks[-1].norm1]
    
    #  配合 timm 與 ViT 的 Reshape
    def reshape_transform(tensor):
        # 去掉 CLS token (index 0)，只取後面的 Patch tokens
        result = tensor[:, 1:, :] 
        
        # 計算長寬 (例如 14x14 patches)
        height = width = int(result.size(1) ** 0.5) 
        
        # [Batch, Tokens, Dim] -> [Batch, Dim, Tokens] -> [Batch, Dim, H, W]
        result = result.transpose(2, 1).transpose(1, 0).reshape(1, result.size(2), height, width)
        return result

    try:
        # 初始化 GradCAM
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        targets = [ClassifierOutputTarget(target_idx)]

        # 如果是半精度模型，input 也要是 half
        if next(model.parameters()).dtype == torch.float16:
             input_tensor = input_tensor.half()

        # 產生熱力圖 (384x384)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # --- 1. 製作原本的疊圖 (Overlay) ---
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # 自適應亮度遮罩 (Adaptive Masking)
        # 避免在全黑背景上顯示熱力圖
        src_img_uint8 = (rgb_img_float * 255).astype(np.uint8)
        gray = cv2.cvtColor(src_img_uint8, cv2.COLOR_RGB2GRAY)
        # 亮度 > 10 才顯示 CAM
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_rgb = cv2.merge([mask, mask, mask])
        final_vis = cv2.bitwise_and(visualization, mask_rgb)
        # 先定義 overlay_bytes並使用不同的變數名稱避免與下方的混淆
        is_overlay_success, overlay_buf = cv2.imencode(".jpg", cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
        overlay_bytes = overlay_buf.tobytes() if is_overlay_success else None
        
        # --- 2. 製作 Resize 回原圖尺寸的純熱力圖 (Raw Heatmap) ---
        raw_heatmap_bytes = None
        if restore_info is not None:
            full_w, full_h = restore_info['full_shape']
            y1, y2, x1, x2 = restore_info['crop_coords']
            
            # 1. 計算裁切後的寬高
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            # 2. 將 384x384 的 CAM Resize 到 "裁切後尺寸" 
            cam_crop_resized = cv2.resize(grayscale_cam, (crop_w, crop_h))
            
            # 3. 建立一個全零的原始尺寸畫布
            full_cam = np.zeros((full_h, full_w), dtype=np.float32)
            
            # 4. 將 Resize 後的 CAM 貼回原本的位置 (Padding Back)
            full_cam[y1:y2, x1:x2] = cam_crop_resized
            
            # 5. 上色 (0~1 -> 0~255 -> JET)
            heatmap_uint8 = (255 * full_cam).astype(np.uint8)
            
            # 如果直接 ApplyColorMap，原本是 0 的地方會變成深藍色 (JET 的 0)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # 將原本 padding 的區域塗黑 (因為 applyColorMap 會把 0 變成藍色)
            # 建立 mask: 1 代表有內容，0 代表 padding
            mask_pad = np.zeros((full_h, full_w), dtype=np.uint8)
            mask_pad[y1:y2, x1:x2] = 255
            # 使用 bitwise_and 只保留中間有值的部分，邊緣變回黑色 (RGB=0,0,0)
            heatmap_color = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask_pad)
            # 編碼純熱力圖
            is_raw_success, raw_buf = cv2.imencode(".jpg", heatmap_color) # heatmap 預設就是 BGR，不需轉換
            if is_raw_success:
                raw_heatmap_bytes = raw_buf.tobytes()

        return overlay_bytes, raw_heatmap_bytes

    except Exception as e:
            print(f"XAI Generation Error: {e}")
            return None, None