import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_xai_image(model, input_tensor, target_idx, rgb_img_float):
    """
    生成 GradCAM 圖片並回傳為 Bytes
    """
    target_layers = [model.backbone.blocks[-1].norm1]

    def reshape_transform(tensor):
        result = tensor[:, 1:, :]
        height = width = int(result.size(1) ** 0.5) 
        result = result.transpose(2, 1).transpose(1, 0).reshape(1, result.size(2), height, width)
        return result

    try:
        # 初始化 GradCAM
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        targets = [ClassifierOutputTarget(target_idx)]

        # 產生熱力圖
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # 疊加
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

        # === 自適應亮度遮罩 (Adaptive Masking) ===
        src_img_uint8 = (rgb_img_float * 255).astype(np.uint8)
        gray = cv2.cvtColor(src_img_uint8, cv2.COLOR_RGB2GRAY)
        
        # 亮度 > 10 才顯示 CAM
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_rgb = cv2.merge([mask, mask, mask])
        
        final_vis = cv2.bitwise_and(visualization, mask_rgb)
        
        # 轉為 Bytes 回傳
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
        if is_success:
            return buffer.tobytes()
        else:
            return None

    except Exception as e:
        print(f"XAI Generation Error: {e}")
        # 出錯時回傳原圖
        src_img_uint8 = (rgb_img_float * 255).astype(np.uint8)
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(src_img_uint8, cv2.COLOR_RGB2BGR))
        return buffer.tobytes()