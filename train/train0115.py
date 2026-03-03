import os, math, time, argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F 

import timm
from timm.utils import ModelEmaV2
from sklearn.metrics import roc_auc_score, precision_recall_curve

# -------------------------------------------------------------------------
# 1. 前處理: 裁切後拉伸 & 安全版 CLAHE
# -------------------------------------------------------------------------
def preprocess_stretch_to_fit(img_arr, tol=7):
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

def apply_conservative_clahe(img_rgb):
    """
    [安全版 CLAHE]
    clipLimit=1.8 (保守值): 能提亮微血管瘤(DR)，又不至於消除輕微白內障的霧感(Cataract)
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

# -------------------------------------------------------------------------
# 2. Loss Function: Asymmetric Loss (AUC Optimized)
# -------------------------------------------------------------------------
class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos  # 設為 0 以最大化 AUC
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        x_sigmoid = torch.clamp(x_sigmoid, min=self.eps, max=1-self.eps)
        
        # --- 正樣本 (Positives) ---
        # gamma_pos=0: 不衰減，讓模型拼命學正樣本 -> 提升 Recall & AUC
        pt0 = x_sigmoid
        loss_pos = -1 * y * ((1 - pt0) ** self.gamma_pos) * torch.log(pt0)
        
        # --- 負樣本 (Negatives) ---
        # gamma_neg=2: 適度壓制背景 -> 提升 Precision (若誤判仍高可改為 4)
        pt1 = 1 - x_sigmoid
        pt1 = (pt1 + self.clip).clamp(max=1) 
        loss_neg = -1 * (1 - y) * ((1 - pt1) ** self.gamma_neg) * torch.log(pt1)
        
        return (loss_pos + loss_neg).mean()

# -------------------------------------------------------------------------
# 3. Dataset (含 Robust Cleaning & Random CLAHE)
# -------------------------------------------------------------------------
class MultiLabelFundusDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, img_size: int = 384, 
                 is_train: bool = True, missing_policy: str = "skip"):
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns and "filename" in self.df.columns:
             self.df["image_path"] = self.df["filename"]
        
        all_y_cols = [c for c in self.df.columns if c.startswith("y_")]
        self.label_cols = [c for c in all_y_cols if "normal" not in c.lower()]
        
        # 數據強健性處理
        self.df[self.label_cols] = self.df[self.label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        print(f"[{'Train' if is_train else 'Val'}] Active Disease Labels: {self.label_cols}")

        if is_train:
            self.tf_augment = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size), 
                    scale=(0.7, 1.0), 
                    ratio=(0.8, 1.25), 
                    interpolation=transforms.InterpolationMode.BICUBIC 
                ),
                transforms.RandomAffine(
                    degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=0 
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.01)
                ], p=0.3),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.6))
                ], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.tf_augment = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        self.valid_indices = []
        all_paths = self.df["image_path"].tolist()
        for idx, rel_path in enumerate(all_paths):
            full_path = self.resolve_image_path(str(rel_path))
            if full_path:
                self.valid_indices.append((idx, full_path))
            elif missing_policy == "skip":
                continue

    def resolve_image_path(self, rel_path: str):
        p1 = os.path.join(self.data_root, rel_path)
        if os.path.exists(p1): return p1
        if rel_path.startswith("images/") or rel_path.startswith("images\\"):
            p2 = os.path.join(self.data_root, rel_path.replace("images/", "", 1).replace("images\\", "", 1))
            if os.path.exists(p2): return p2
        p3 = os.path.join(self.data_root, "images", rel_path)
        if os.path.exists(p3): return p3
        return None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        original_idx, full_path = self.valid_indices[index]
        row = self.df.iloc[original_idx]
        
        img_arr = None
        img_cv = cv2.imread(full_path)
        if img_cv is not None:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_arr = preprocess_stretch_to_fit(img_cv)
            
            # [關鍵修正：安全版 CLAHE 策略]
            # 1. 訓練時：50% 機率使用，讓模型學習適應兩種風格，保護白內障特徵
            if self.is_train and np.random.rand() < 0.5:
                 img_arr = apply_conservative_clahe(img_arr)
            # 2. 驗證/測試時：為了對抗陌生資料的光照差異，建議強制開啟
            elif not self.is_train: 
                 img_arr = apply_conservative_clahe(img_arr)
        
        if img_arr is None:
            img_arr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_pil = Image.fromarray(img_arr)
        x = self.tf_augment(img_pil)
        y = row[self.label_cols].to_numpy(dtype=np.float32, copy=True)
        return x, torch.from_numpy(y)

# -------------------------------------------------------------------------
# 4. Model
# -------------------------------------------------------------------------
class ViTMultiLabel(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features * 2, num_labels)
        
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        cls_token = feats[:, 0]
        gap_feat = feats[:, 1:].mean(dim=1)
        combined_feat = torch.cat([cls_token, gap_feat], dim=1)
        return self.head(combined_feat)

def get_param_groups(model, base_lr, weight_decay):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "head" in name: head_params.append(param)
        else: backbone_params.append(param)
    print(f"Layer Decay Setup: Head LR={base_lr}, Backbone LR={base_lr * 0.1:.2e}")
    return [
        {'params': head_params, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': backbone_params, 'lr': base_lr * 0.1, 'weight_decay': weight_decay}
    ]

def safe_torch_load(path: str):
    import argparse as _argparse
    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception: pass
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")

def load_retfound_weights_into_vit(vit: nn.Module, ckpt_path: str):
    print(f"Loading weights from {ckpt_path} ...")
    ckpt = safe_torch_load(ckpt_path)
    sd = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "backbone.", "encoder.", "student.", "model."):
            if nk.startswith(prefix): nk = nk[len(prefix):]
        new_sd[nk] = v
    if 'pos_embed' in new_sd:
        ckpt_pos = new_sd['pos_embed']
        model_pos = vit.pos_embed
        if ckpt_pos.shape != model_pos.shape:
            print(f"Resize pos_embed: {ckpt_pos.shape} -> {model_pos.shape}")
            grid = ckpt_pos[:, 1:, :]
            gs_old = int(math.sqrt(grid.shape[1]))
            gs_new = int(math.sqrt(model_pos.shape[1]-1))
            grid = grid.permute(0, 2, 1).reshape(1, -1, gs_old, gs_old)
            grid = F.interpolate(grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
            grid = grid.flatten(2).permute(0, 2, 1)
            new_sd['pos_embed'] = torch.cat((ckpt_pos[:, 0:1, :], grid), dim=1)
    vit.load_state_dict(new_sd, strict=False)

def build_model(num_labels, ckpt_path="", backbone="vit_large_patch16_224", img_size=384):
    vit = timm.create_model(backbone, pretrained=False, num_classes=0, img_size=img_size)
    if ckpt_path: load_retfound_weights_into_vit(vit, ckpt_path)
    model = ViTMultiLabel(vit, num_labels)
    return model

# -------------------------------------------------------------------------
# 5. Training & Evaluation
# -------------------------------------------------------------------------
def find_optimal_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_thresh, f1_scores[best_idx]

@torch.no_grad()
def evaluate(model, loader, device, label_cols, criterion):
    model.eval()
    all_y, all_p = [], []
    total_loss, n = 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y) 
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_p.append(torch.sigmoid(logits).float().cpu().numpy())
        all_y.append(y.cpu().numpy())

    all_p = np.concatenate(all_p, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    
    aucs = []
    best_thresholds = {}
    
    for j in range(all_y.shape[1]):
        yj = all_y[:, j]
        pj = all_p[:, j]
        label_name = label_cols[j]
        if len(np.unique(yj)) < 2: auc = 0.5
        else: auc = roc_auc_score(yj, pj)
        aucs.append(auc)
        best_thr, best_f1 = find_optimal_threshold(yj, pj)
        best_thresholds[label_name] = {"thresh": round(best_thr, 3), "f1": round(best_f1, 3)}

    macro_auc = float(np.mean(aucs)) if len(aucs) else 0.0
    return total_loss / max(n, 1), macro_auc, aucs, best_thresholds

def train_one_epoch(model, loader, optimizer, device, scaler, grad_accum_steps, model_ema, criterion):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running, n = 0.0, 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        y_smooth = y * 0.9 + 0.05 

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss_full = criterion(logits, y_smooth)
            loss_scaled = loss_full / grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            if model_ema is not None:
                model_ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        running += loss_full.item() * x.size(0) 
        n += x.size(0)

    if (step % grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None:
            model_ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    return running / max(n, 1)

# -------------------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_root", type=str, default="/content/drive/MyDrive/fundus-retfound-data")
    ap.add_argument("--local_root", type=str, default="/content/local_images")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/Model_outputs")
    ap.add_argument("--train_csv", type=str, default="")
    ap.add_argument("--val_csv", type=str, default="")
    
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=20) 
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5) 
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ---------------------------------------------------------------------
    # [新增] 強制路徑檢查，防止跑錯資料
    # ---------------------------------------------------------------------
    if not args.data_root:
        if args.local_root and os.path.exists(args.local_root):
            args.data_root = args.local_root
            print(f"Using local_root as data_root: {args.data_root}")
        else:
            raise ValueError("Error: Must provide --data_root or a valid --local_root.")
            
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Error: Image directory not found: {args.data_root}")

    if not args.train_csv or not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Error: Train CSV not found: '{args.train_csv}'")
    if not args.val_csv or not os.path.exists(args.val_csv):
        raise FileNotFoundError(f"Error: Val CSV not found: '{args.val_csv}'")
    # ---------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = MultiLabelFundusDataset(args.train_csv, args.data_root, args.img_size, True)
    val_ds = MultiLabelFundusDataset(args.val_csv, args.data_root, args.img_size, False)
    label_cols = train_ds.label_cols
    print(f"Labels: {label_cols}")
    
    # 使用 ASL (Optimized for AUC)
    criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0, clip=0.05).to(device)
    print("Using Asymmetric Loss Optimized (gamma_neg=2, gamma_pos=0)")

    train_loader = DataLoader(train_ds, args.batch_size, True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, False, num_workers=2, pin_memory=True)
    
    ckpt_path = args.ckpt_path or os.path.join(args.drive_root, "RETFound_mae_natureCFP.pth")
    model = build_model(len(label_cols), ckpt_path, img_size=args.img_size).to(device)
    model_ema = ModelEmaV2(model, decay=0.99, device=device)
    
    optimizer_params = get_param_groups(model, base_lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(optimizer_params)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    
    start_epoch = 1 
    best_auc = -1.0

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_ema.module.load_state_dict(checkpoint['model_ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', -1.0)
        print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    
    print(f"Start Training (Epochs={args.epochs}, Start={start_epoch})...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, 
                                     args.grad_accum, model_ema, criterion)
        
        val_loss, val_auc, label_aucs, best_thr = evaluate(model_ema.module, val_loader, device, label_cols, criterion)
        dt = time.time() - t0
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Time: {dt:.1f}s")
        print(f"   AUCs: {dict(zip(label_cols, np.round(label_aucs, 3)))}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.out_dir, "best_ema_RETFound.pth")
            torch.save({
                "model": model_ema.module.state_dict(),
                "best_auc": best_auc,
                "epoch": epoch,
                "labels": label_cols,
                "suggested_thresholds": best_thr 
            }, save_path)
            print(f"   --> Saved Best EMA Model to {save_path}")

        ckpt_save_path = os.path.join(args.out_dir, "checkpoint_optimizer.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_ema_state_dict": model_ema.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_auc": best_auc,
        }, ckpt_save_path)

if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    main()