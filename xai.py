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
        
        # 預設回傳 None，相容 controller 的接收格式，並跳過疊圖處理
        overlay_bytes = None 
        raw_heatmap_bytes = None
        
        if restore_info is not None:
            full_w, full_h = restore_info['full_shape']
            y1, y2, x1, x2 = restore_info['crop_coords']
            
            # 1. 計算裁切後的寬高
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            # 2. 將 384x384 的 CAM Resize 回"裁切後尺寸"，並貼回全尺寸畫布
            cam_crop_resized = cv2.resize(grayscale_cam, (crop_w, crop_h))
            full_cam = np.zeros((full_h, full_w), dtype=np.float32)
            full_cam[y1:y2, x1:x2] = cam_crop_resized
            
            # 3. 將 384x384 的 RGB 影像也 Resize 回"裁切後尺寸"，並貼回全尺寸畫布
            src_img_uint8 = (rgb_img_float * 255).astype(np.uint8)
            img_crop_resized = cv2.resize(src_img_uint8, (crop_w, crop_h))
            full_img = np.zeros((full_h, full_w, 3), dtype=np.uint8)
            full_img[y1:y2, x1:x2] = img_crop_resized
            
            # 4. 在「全尺寸畫布」上計算 MASK 座標
            gray = cv2.cvtColor(full_img, cv2.COLOR_RGB2GRAY)
            # 亮度 > 10 視為有效區域 (過濾掉 Padding 與原圖的黑色背景)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            # 5. 將全尺寸 CAM 上色
            heatmap_uint8 = (255 * full_cam).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # 6. 套用遮罩：保留有內容的區域，邊緣 Padding 與極黑背景變回黑色
            final_heatmap = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)
            
            # 7. 編碼為 JPG
            is_raw_success, raw_buf = cv2.imencode(".jpg", final_heatmap)
            if is_raw_success:
                raw_heatmap_bytes = raw_buf.tobytes()
                
        else:
            # 容錯處理：若無 restore_info，則以輸入尺寸計算
            src_img_uint8 = (rgb_img_float * 255).astype(np.uint8)
            gray = cv2.cvtColor(src_img_uint8, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            heatmap_uint8 = (255 * grayscale_cam).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            final_heatmap = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)
            
            is_raw_success, raw_buf = cv2.imencode(".jpg", final_heatmap)
            if is_raw_success:
                raw_heatmap_bytes = raw_buf.tobytes()

        # 回傳 None 與純熱力圖的 bytes，完美銜接外部呼叫
        return overlay_bytes, raw_heatmap_bytes

    except Exception as e:
            print(f"XAI Generation Error: {e}")
            return None, None