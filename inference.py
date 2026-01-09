import numpy as np

# 定義標籤與閾值
LABELS = ['normal', 'dr', 'glaucoma', 'cataract']
CLASS_THRESHOLDS = {
    'normal': 0.60,
    'dr': 0.50,
    'glaucoma': 0.45,
    'cataract': 0.45 
}
MARGIN_THRESHOLD = 0.3
RISK_TARGET = 0.3
NORMAL_LOW = 0.3

def analyze_results(probs):
    """
    根據機率分析疾病風險
    Input: probs (numpy array)
    Output: dict (JSON-ready structure)
    """
    results = {}
    
    # 1. 找出 Normal Index
    normal_idx = -1
    for i, l in enumerate(LABELS):
        if 'normal' in l.lower(): 
            normal_idx = i
            break
            
    prob_normal = float(probs[normal_idx]) if normal_idx != -1 else 0.0
    
    # 2. 排序所有疾病
    all_diseases = []
    for i, label in enumerate(LABELS):
        if i == normal_idx: continue
        all_diseases.append({
            "index": i, 
            "label": label, 
            "prob": float(probs[i])
        })
    
    all_diseases.sort(key=lambda x: x['prob'], reverse=True)
    top_disease = all_diseases[0]
    
    # 3. 過濾出 Active Diseases (超過閾值)
    active_diseases = []
    for d in all_diseases:
        thresh = CLASS_THRESHOLDS.get(d['label'], 0.5)
        if d['prob'] > thresh:
            active_diseases.append(d)
            
    # 4. 決策邏輯 (Diagnosis Generation)
    diff = prob_normal - top_disease['prob']
    diagnosis_status = "Unknown"
    diagnosis_text = ""
    target_cam_idx = np.argmax(probs) # 預設 CAM 看機率最高的
    risks_list = []
    
    # --- 分支一：Normal ---
    if diff >= MARGIN_THRESHOLD:
        diagnosis_status = "Normal"
        diagnosis_text = f"Diagnosis: Normal (Conf: {prob_normal:.1%})"
        target_cam_idx = normal_idx
        
        # 紀錄被忽略的微量病徵
        if active_diseases:
            ignored = ", ".join([f"{d['label']}({d['prob']:.1%})" for d in active_diseases])
            diagnosis_text += f"\n[Note] Ignored: {ignored}"
            
    # --- 分支二：異常 (Abnormal) ---
    else:
        diagnosis_status = "Abnormal"
        
        # A. 多重疾病
        if len(active_diseases) > 1:
            diagnosis_text = "Multiple Detected:\n" + "\n".join([f"• {d['label']} ({d['prob']:.1%})" for d in active_diseases])
            target_cam_idx = active_diseases[0]['index']
            
            # 檢查剩下的 Risk
            remaining = [d for d in all_diseases if d not in active_diseases]
            risks_list = [d for d in remaining if d['prob'] > RISK_TARGET]
            
        # B. 單一主要疾病
        else:
            diagnosis_text = f"Detected: {top_disease['label']} ({top_disease['prob']:.1%})"
            target_cam_idx = top_disease['index']
            
            # Risk 邏輯
            if prob_normal < NORMAL_LOW:
                remaining = [d for d in all_diseases if d != top_disease]
                candidates = [d for d in remaining if d['prob'] > RISK_TARGET]
                
                if candidates:
                    risks_list = candidates
                elif remaining:
                    risks_list = [remaining[0]] # 強制取下一個最高的當 Risk
        
        # 附加 Risk 文字
        if risks_list:
            diagnosis_text += "\n⚠️ Risk / Co-morbidity:\n" + "\n".join([f"• {r['label']} ({r['prob']:.1%})" for r in risks_list])

    # 5. 回傳結構化資料
    return {
        "status": diagnosis_status,
        "diagnosis_text": diagnosis_text,
        "probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "target_cam_idx": int(target_cam_idx),
        "is_dr": any(d['label'] == 'dr' for d in active_diseases) or (top_disease['label'] == 'dr' and diagnosis_status == "Abnormal")
    }