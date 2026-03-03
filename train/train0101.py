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

import timm
from sklearn.metrics import roc_auc_score

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


def crop_image_from_gray(img_arr, tol=7):
    """
    OpenCV 自動去黑邊邏輯
    """
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
            img1 = img_arr[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img_arr[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img_arr[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)
    return img_arr


class MultiLabelFundusDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, img_size: int = 224, 
                 is_train: bool = True, missing_policy: str = "skip"):
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        self.missing_policy = missing_policy
        
        # 1. 讀取 CSV
        self.df = pd.read_csv(csv_path)
        
        if "image_path" not in self.df.columns:
             if "filename" in self.df.columns:
                 self.df["image_path"] = self.df["filename"]
        
        # 自動抓取標籤 (確保順序與計算權重時一致: Normal, DR, Glaucoma, Cataract)
        # 為了安全，這裡建議強制排序，或是依賴 CSV 原始順序 (通常 Pandas 讀取順序是固定的)
        self.label_cols = [c for c in self.df.columns if c.startswith("y_")]
        assert len(self.label_cols) > 0, f"No y_* label cols in {csv_path}"

        # 2. 定義 Augmentation
        if is_train:
            self.tf_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.tf_augment = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        # 3. 執行路徑檢查與過濾
        print(f"[{'Train' if is_train else 'Val'}] Checking image paths with policy='{missing_policy}'...")
        self.valid_indices = []
        
        all_paths = self.df["image_path"].tolist()
        missing_count = 0
        
        for idx, rel_path in enumerate(all_paths):
            full_path = self.resolve_image_path(str(rel_path))
            
            if full_path is None:
                missing_count += 1
                if missing_policy == "raise":
                    raise FileNotFoundError(f"Image not found: {rel_path} in {data_root}")
                elif missing_policy == "skip":
                    continue
                elif missing_policy == "black":
                    self.valid_indices.append((idx, None))
            else:
                self.valid_indices.append((idx, full_path))

        if missing_policy == "skip" and missing_count > 0:
            print(f"  ⚠️ Skipped {missing_count} missing images. Remaining: {len(self.valid_indices)}")
        elif missing_count == 0:
            print(f"  ✅ All {len(self.valid_indices)} images found.")
        else:
            print(f"  ⚠️ Warning: {missing_count} images missing but policy is '{missing_policy}'.")

    def resolve_image_path(self, rel_path: str):
        # 智慧路徑解析
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
                img_arr = crop_image_from_gray(img_cv)
        
        if img_arr is None:
            img_arr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_pil = Image.fromarray(img_arr)
        img_pil = img_pil.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        x = self.tf_augment(img_pil)

        y = row[self.label_cols].to_numpy(dtype=np.float32, copy=True)
        y = torch.from_numpy(y)
        return x, y


class ViTMultiLabel(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.num_features
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        if feats.dim() == 3:
            # === [重大優化] 改用 GAP (Global Average Pooling) ===
            # 移除 [CLS] token (index 0)，對剩下的 Patch 做平均
            # 這能捕捉全圖特徵，對視神經盤等細節更佳
            feats = feats[:, 1:].mean(dim=1) 
        return self.head(feats)


def safe_torch_load(path: str):
    import argparse as _argparse
    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([_argparse.Namespace])
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_retfound_weights_into_vit(vit: nn.Module, ckpt_path: str):
    print(f"Loading RETFound weights from {ckpt_path} ...")
    ckpt = safe_torch_load(ckpt_path)

    if isinstance(ckpt, dict):
        if "model" in ckpt: sd = ckpt["model"]
        elif "state_dict" in ckpt: sd = ckpt["state_dict"]
        else: sd = ckpt
    else:
        raise ValueError("Unexpected checkpoint format.")

    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "backbone.", "encoder.", "student.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_sd[nk] = v

    missing, unexpected = vit.load_state_dict(new_sd, strict=False)
    print(f"Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def build_model(num_labels: int, ckpt_path: str = "", freeze_backbone: bool = False, backbone_name: str = "vit_large_patch16_224"):
    vit = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    if ckpt_path:
        load_retfound_weights_into_vit(vit, ckpt_path)
    model = ViTMultiLabel(vit, num_labels=num_labels)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen.")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
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
    for j in range(all_y.shape[1]):
        yj = all_y[:, j]
        pj = all_p[:, j]
        if len(np.unique(yj)) < 2:
            aucs.append(0.5) 
        else:
            aucs.append(roc_auc_score(yj, pj))
    
    macro_auc = float(np.mean(aucs)) if len(aucs) else 0.0
    return total_loss / max(n, 1), macro_auc, aucs


def train_one_epoch(model, loader, optimizer, device, scaler, grad_accum_steps: int):
    model.train()
    
    # === [重大優化] 加入類別平衡權重 (Pos Weight) ===
    # 數值來源: 您提供的計算結果
    # 對應順序: [y_normal, y_dr, y_glaucoma, y_cataract]
    # 請務必確認您的 labels 順序是否一致 (通常是這個順序)
    class_weights = torch.tensor([1.6901, 2.6558, 6.4109, 17.2095]).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    optimizer.zero_grad(set_to_none=True)
    running, n = 0.0, 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y) / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * x.size(0) * grad_accum_steps
        n += x.size(0)

    if (step % grad_accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running / max(n, 1)


def main():
    ap = argparse.ArgumentParser()

    # 1. 定義預設變數
    default_out_dir = "/content/drive/MyDrive/Model_outputs"
    # 移除 cfp 層級，依據您的最新結構
    default_drive_root = "/content/drive/MyDrive/fundus-retfound-data" 

    # 路徑設定
    ap.add_argument("--drive_root", type=str, default=default_drive_root)
    ap.add_argument("--local_root", type=str, default="/content/local_images")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--train_csv", type=str, default="")
    ap.add_argument("--val_csv", type=str, default="")
    ap.add_argument("--out_dir", type=str, default=default_out_dir)
    ap.add_argument("--missing_policy", type=str, default="skip", choices=["raise", "skip", "black"])

    # 訓練超參數
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--freeze_backbone", action="store_true")

    # 模型與權重
    ap.add_argument("--backbone", type=str, default="vit_large_patch16_224")
    ap.add_argument("--hf_repo", type=str, default="YukunZhou/RETFound_mae_natureCFP")
    ap.add_argument("--hf_filename", type=str, default="RETFound_mae_natureCFP.pth")
    ap.add_argument("--ckpt_path", type=str, default="")
    
    args, _ = ap.parse_known_args()

    if "/content/drive" in args.out_dir and not os.path.exists("/content/drive"):
        print("⚠️ 警告：偵測到輸出路徑在 Drive，但似乎未掛載。")
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"📁 模型將儲存於: {args.out_dir}")

    drive_root = args.drive_root
    local_root = args.local_root
    
    # 決定資料路徑
    if not args.data_root:
        if os.path.exists(local_root):
            print(f"Found local images at {local_root}, using it.")
            args.data_root = local_root
        else:
            fallback_path = os.path.join(drive_root, "images")
            print(f"Local images not found, falling back to Drive: {fallback_path}")
            args.data_root = fallback_path

    # CSV 路徑 (無 cfp 層)
    if not args.train_csv:
        args.train_csv = os.path.join(drive_root, "labels", "train_clean.csv")
    if not args.val_csv:
        args.val_csv = os.path.join(drive_root, "labels", "val_clean.csv")

    print(f"Data Root: {args.data_root}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV  : {args.val_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 權重檔
    ckpt_path = args.ckpt_path.strip()
    if not ckpt_path:
        possible_ckpt = os.path.join(drive_root, "RETFound_mae_natureCFP.pth")
        if os.path.exists(possible_ckpt):
            ckpt_path = possible_ckpt

    if not ckpt_path and hf_hub_download is not None:
        try:
            token = os.environ.get("HF_TOKEN")
            ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename, token=token)
        except Exception as e:
            print(f"Download failed: {e}. Please manually set --ckpt_path")

    # 讀取標籤並檢查順序
    tmp_df = pd.read_csv(args.train_csv)
    label_cols = [c for c in tmp_df.columns if c.startswith("y_")]
    print(f"Labels ({len(label_cols)}): {label_cols}")
    
    # 防呆檢查：確認 Label 順序是否如預期 (Normal, DR, Glaucoma, Cataract)
    expected_order = ['y_normal', 'y_dr', 'y_glaucoma', 'y_cataract']
    if label_cols == expected_order:
        print("✅ Label order matches expected weights.")
    else:
        print("⚠️ Warning: Label order differs! Please check pos_weight manually.")
        print(f"   Expected: {expected_order}")
        print(f"   Actual  : {label_cols}")

    # 建立 Dataset
    train_ds = MultiLabelFundusDataset(args.train_csv, args.data_root, img_size=args.img_size, 
                                       is_train=True, missing_policy=args.missing_policy)
    val_ds   = MultiLabelFundusDataset(args.val_csv,   args.data_root, img_size=args.img_size, 
                                       is_train=False, missing_policy=args.missing_policy)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True)

    # 建立模型
    model = build_model(num_labels=len(label_cols), ckpt_path=ckpt_path, 
                        freeze_backbone=args.freeze_backbone, backbone_name=args.backbone).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_auc = -1.0
    
    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, args.grad_accum)
        val_loss, val_auc, label_aucs = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro AUC: {val_auc:.4f} | Time: {dt:.1f}s")
        # 顯示各類別 AUC，方便觀察青光眼與白內障是否有進步
        auc_dict = dict(zip(label_cols, np.round(label_aucs, 3)))
        print(f"   Details AUC: {auc_dict}")

        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.out_dir, "best_model.pth")
            # 儲存 labels 順序以供推論使用
            torch.save({"model": model.state_dict(), "best_auc": best_auc, "epoch": epoch, "labels": label_cols}, save_path)
            print(f"   --> Saved Best Model: {save_path}")

    print(f"Done. Best Macro AUC: {best_auc:.4f}")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()