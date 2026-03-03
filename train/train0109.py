import os, math, time, argparse
from pathlib import Path
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
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# --- [前處理] 裁切後拉伸 (Stretch to Fit) ---
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

class MultiLabelFundusDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, img_size: int = 224, 
                 is_train: bool = True, missing_policy: str = "skip"):
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        self.missing_policy = missing_policy
        
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns:
             if "filename" in self.df.columns:
                 self.df["image_path"] = self.df["filename"]
        
        self.label_cols = [c for c in self.df.columns if c.startswith("y_")]
        assert len(self.label_cols) > 0, f"No y_* label cols in {csv_path}"

        # 增強策略 (V3.0: 幾何強力增強)
        if is_train:
            self.tf_augment = transforms.Compose([
                # 隨機裁切縮放 (模擬特寫)
                transforms.RandomResizedCrop(
                    size=(self.img_size, self.img_size), 
                    scale=(0.6, 1.0), 
                    ratio=(0.85, 1.15),
                    interpolation=transforms.InterpolationMode.BICUBIC # 這裡也建議顯式指定
                ),
                # 仿射變換 (模擬變形)
                transforms.RandomAffine(
                    degrees=15,          
                    translate=(0.05, 0.05), 
                    scale=(0.95, 1.05), 
                    shear=5              
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
                ], p=0.3),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))
                ], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            # === [修正點] ===
            # 將 resample 改為 interpolation
            # 並使用 transforms.InterpolationMode.BICUBIC
            self.tf_augment = transforms.Compose([
                transforms.Resize(
                    (self.img_size, self.img_size), 
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        print(f"[{'Train' if is_train else 'Val'}] Checking image paths...")
        self.valid_indices = []
        all_paths = self.df["image_path"].tolist()
        missing_count = 0
        
        for idx, rel_path in enumerate(all_paths):
            full_path = self.resolve_image_path(str(rel_path))
            if full_path is None:
                missing_count += 1
                if missing_policy == "skip": continue
                elif missing_policy == "black": self.valid_indices.append((idx, None))
            else:
                self.valid_indices.append((idx, full_path))
        print(f"  Valid images: {len(self.valid_indices)} (Skipped: {missing_count})")

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
        if full_path:
            img_cv = cv2.imread(full_path)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_arr = preprocess_stretch_to_fit(img_cv)
        
        if img_arr is None:
            img_arr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_pil = Image.fromarray(img_arr)
        x = self.tf_augment(img_pil)
        y = row[self.label_cols].to_numpy(dtype=np.float32, copy=True)
        return x, torch.from_numpy(y)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        original_idx, full_path = self.valid_indices[index]
        row = self.df.iloc[original_idx]
        
        img_arr = None
        if full_path:
            img_cv = cv2.imread(full_path)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_arr = preprocess_stretch_to_fit(img_cv)
        
        if img_arr is None:
            img_arr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_pil = Image.fromarray(img_arr)
        x = self.tf_augment(img_pil)
        y = row[self.label_cols].to_numpy(dtype=np.float32, copy=True)
        return x, torch.from_numpy(y)

class ViTMultiLabel(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features, num_labels)
    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3: feats = feats[:, 1:].mean(dim=1)
        return self.head(feats)

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

def build_model(num_labels, ckpt_path="", freeze_backbone=False, backbone="vit_large_patch16_224", img_size=224):
    vit = timm.create_model(backbone, pretrained=False, num_classes=0, img_size=img_size)
    if ckpt_path: load_retfound_weights_into_vit(vit, ckpt_path)
    model = ViTMultiLabel(vit, num_labels)
    if freeze_backbone:
        for p in model.backbone.parameters(): p.requires_grad = False
    return model

# --- [新增功能] 自動尋找最佳 F1 門檻值 ---
def find_optimal_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    # thresholds 陣列比 precision/recall 少一個，所以要小心索引
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return best_thresh, best_f1

@torch.no_grad()
def evaluate(model, loader, device, label_cols):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_y, all_p = [], []
    total_loss, n = 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_p.append(torch.sigmoid(logits).float().cpu().numpy())
        all_y.append(y.cpu().numpy())

    all_p = np.concatenate(all_p, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    
    aucs = []
    best_thresholds = {}
    
    # 計算每個類別的 AUC 和 最佳 F1 門檻
    for j in range(all_y.shape[1]):
        yj = all_y[:, j]
        pj = all_p[:, j]
        label_name = label_cols[j]
        
        # AUC
        if len(np.unique(yj)) < 2: auc = 0.5
        else: auc = roc_auc_score(yj, pj)
        aucs.append(auc)
        
        # Optimal Threshold
        best_thr, best_f1 = find_optimal_threshold(yj, pj)
        best_thresholds[label_name] = {"thresh": round(best_thr, 3), "f1": round(best_f1, 3)}

    macro_auc = float(np.mean(aucs)) if len(aucs) else 0.0
    return total_loss / max(n, 1), macro_auc, aucs, best_thresholds

def train_one_epoch(model, loader, optimizer, device, scaler, grad_accum, model_ema):
    model.train()
    # 手動設定權重 (基於資料比例)
    class_weights = torch.tensor([1.0, 1.0, 3.0, 6.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    optimizer.zero_grad(set_to_none=True)
    running, n = 0.0, 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        y_smooth = y * 0.9 + 0.05 

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y_smooth) / grad_accum

        scaler.scale(loss).backward()

        if step % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            if model_ema: model_ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * x.size(0) * grad_accum
        n += x.size(0)

    if (step % grad_accum) != 0:
        scaler.step(optimizer)
        scaler.update()
        if model_ema: model_ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    return running / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive_root", type=str, default="/content/drive/MyDrive/fundus-retfound-data")
    ap.add_argument("--local_root", type=str, default="/content/local_images")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/Model_outputs")
    ap.add_argument("--train_csv", type=str, default="")
    ap.add_argument("--val_csv", type=str, default="")
    
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--ckpt_path", type=str, default="")
    
    args, _ = ap.parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 路徑解析
    if not args.data_root:
        args.data_root = args.local_root if os.path.exists(args.local_root) else os.path.join(args.drive_root, "images")
    if not args.train_csv: args.train_csv = os.path.join(args.drive_root, "labels", "train_clean.csv")
    if not args.val_csv: args.val_csv = os.path.join(args.drive_root, "labels", "val_clean.csv")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset & Label Detection
    tmp_df = pd.read_csv(args.train_csv)
    label_cols = [c for c in tmp_df.columns if c.startswith("y_")]
    print(f"Labels: {label_cols}")
    
    train_ds = MultiLabelFundusDataset(args.train_csv, args.data_root, args.img_size, True)
    val_ds = MultiLabelFundusDataset(args.val_csv, args.data_root, args.img_size, False)
    
    train_loader = DataLoader(train_ds, args.batch_size, True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, False, num_workers=2, pin_memory=True)
    
    # Model
    ckpt_path = args.ckpt_path or os.path.join(args.drive_root, "RETFound_mae_natureCFP.pth")
    model = build_model(len(label_cols), ckpt_path, img_size=args.img_size).to(device)
    model_ema = ModelEmaV2(model, decay=0.99, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    
    best_auc = -1.0
    print("Start Training V3.1 (Optimal Thresholds + Manual Weights)...")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, args.grad_accum, model_ema)
        # 傳入 label_cols 以便計算 per-label threshold
        val_loss, val_auc, label_aucs, best_thr = evaluate(model_ema.module, val_loader, device, label_cols)
        dt = time.time() - t0
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Time: {dt:.1f}s")
        print(f"   AUCs: {dict(zip(label_cols, np.round(label_aucs, 3)))}")
        print(f"   🔥 Suggested Thresholds: {best_thr}") # 顯示推薦門檻

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save({
                "model": model.state_dict(),
                "best_auc": best_auc,
                "epoch": epoch,
                "labels": label_cols,
                "suggested_thresholds": best_thr # 將最佳門檻存入檔案
            }, save_path)
            print(f"   --> Saved Best Model")

if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    main()