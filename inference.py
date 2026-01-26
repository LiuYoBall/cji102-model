import numpy as np
from cfp_classify import LABELS_LIST  # 引用定義好的標籤

# 雙重閾值設定
THRESHOLDS = {
    'y_dr':       {'high': 0.6, 'low': 0.35},
    'y_glaucoma': {'high': 0.5, 'low': 0.25},
    'y_cataract': {'high': 0.5, 'low': 0.25}
}

def analyze_results(probs):
    """
    新版邏輯：雙重閾值判定 (Confirmed vs Risk)
    Input: probs (numpy array of shape (3,))
    Output: dict (Full JSON analysis)
    """
    
    # 確保維度正確
    if len(probs) != len(LABELS_LIST):
        # 簡單的防呆，避免舊模型配新代碼
        raise ValueError(f"Shape mismatch: Probs {len(probs)} vs Labels {len(LABELS_LIST)}")

    # 1. 拆解機率 (DR:max, Others:mean 在推論層已經做過 Max TTA 了，這裡直接取值)
    # 註：如果你的模型是分別訓練的，這裡邏輯可以調整。目前假設 probs 已經是最終機率。
    final_probs = probs 

    # 2. 演算法核心：獨立收集 High List 與 Risk List
    high_detected = [] 
    risk_detected = [] 
    
    for i, lbl in enumerate(LABELS_LIST):
        t_high = THRESHOLDS[lbl]['high']
        t_low  = THRESHOLDS[lbl]['low']
        prob = float(final_probs[i])
        
        # 顯示名稱去前綴
        display_name = lbl.replace('y_', '').upper()
        
        item = {
            "name": display_name,
            "prob": prob,
            "index": i,
            "key": lbl # 前端可能需要原始 key
        }
        
        if prob >= t_high:
            high_detected.append(item)
        elif prob >= t_low:
            risk_detected.append(item)
            
    # 排序 (機率高到低)
    high_detected.sort(key=lambda x: x['prob'], reverse=True)
    risk_detected.sort(key=lambda x: x['prob'], reverse=True)

    # 3. 狀態判定與文字生成
    status = "Normal Findings"
    diagnosis_lines = []
    target_cam_idx = 0 # 預設 CAM
    
    if len(high_detected) > 0:
        status = "Disease Detected"
        diagnosis_lines.append("Detected (High Confidence):")
        for item in high_detected:
            diagnosis_lines.append(f"• {item['name']} ({item['prob']:.1%})")
            
        if len(risk_detected) > 0:
            diagnosis_lines.append("Risk / Suspicious:")
            for item in risk_detected:
                diagnosis_lines.append(f"• {item['name']} ({item['prob']:.1%})")
                
        # CAM 優先看最嚴重的確診
        target_cam_idx = high_detected[0]['index']
        
    elif len(risk_detected) > 0:
        status = "Risk / Suspicious"
        diagnosis_lines.append("Risk Areas:")
        for item in risk_detected:
            diagnosis_lines.append(f"• {item['name']} ({item['prob']:.1%})")
        
        # CAM 看最可疑的風險
        target_cam_idx = risk_detected[0]['index']
        
    else:
        status = "Normal Findings"
        diagnosis_lines.append("No significant findings observed.")
        diagnosis_lines.append("(All probs < Low Thresholds)")
        # Normal 時，CAM 看機率相對最高的那個地方 (解釋為什麼有一點點機率)
        target_cam_idx = int(np.argmax(final_probs))

    diagnosis_text = "\n".join(diagnosis_lines)

    # 4. 回傳結構化資料
    return {
        "status": status,
        "diagnosis_text": diagnosis_text,
        "high_risk": high_detected,   # 給前端畫紅燈
        "low_risk": risk_detected,    # 給前端畫橘燈
        "all_probs": {LABELS_LIST[i]: float(probs[i]) for i in range(len(LABELS_LIST))},
        "target_cam_idx": int(target_cam_idx),
        "is_dr": any(d['key'] == 'y_dr' for d in high_detected) # 為了保留 YOLO 觸發邏輯
    }