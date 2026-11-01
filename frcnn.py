import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "/arf/scratch/bulas/YL_tez/yolo/Datasets_301225/Datasets/galaxy"

NUM_CLASSES = 3   # background + MG + CG
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
EPOCHS = 50
LR = 1e-4

BATCH_SIZE = 4            
ACCUMULATION_STEPS = 2    
PATIENCE = 20             
SCHEDULER_PATIENCE = 5    
WEIGHT_DECAY = 5e-4       # <-- INCREASED: Stronger regularization to prevent memorization

CLASS_NAMES = {1: "MG", 2: "CG"}

# =====================================================
# DATASET
# =====================================================

class GalaxyDataset(Dataset):
    def __init__(self, root_dir, split):
        self.split = split 
        self.img_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)

        valid_img_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_img_ext)])
        self.labels = sorted([f for f in os.listdir(self.label_dir) if f.lower().endswith('.txt')])
        
        if len(self.images) != len(self.labels):
            print(f"[WARNING] Mismatch in '{split}' split: {len(self.images)} images vs {len(self.labels)} labels!")

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"OpenCV Error: Could not read image at '{img_path}'.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        h, w, _ = img.shape

        with open(label_path) as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                    
                cls, x, y, bw, bh = map(float, line.split())
                x1 = (x - bw/2) * w
                y1 = (y - bh/2) * h
                x2 = (x + bw/2) * w
                y2 = (y + bh/2) * h

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        boxes_array = np.array(boxes, dtype=np.float32)

        # SPATIAL AUGMENTATIONS (Modify bounding boxes)
        if self.split == "train" and len(boxes_array) > 0:
            # 1. Horizontal Flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                boxes_array[:, [0, 2]] = w - boxes_array[:, [2, 0]]
            
            # 2. Vertical Flip
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                boxes_array[:, [1, 3]] = h - boxes_array[:, [3, 1]]
                
            # 3. 90-Degree Clockwise Rotation (NEW)
            if random.random() < 0.5:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                # Math to correctly rotate bounding boxes 90 degrees
                x1_old, y1_old = boxes_array[:, 0].copy(), boxes_array[:, 1].copy()
                x2_old, y2_old = boxes_array[:, 2].copy(), boxes_array[:, 3].copy()
                
                boxes_array[:, 0] = h - y2_old
                boxes_array[:, 1] = x1_old
                boxes_array[:, 2] = h - y1_old
                boxes_array[:, 3] = x2_old
                
                # Swap width and height to match the rotated image dimensions
                h, w = w, h

        boxes = torch.as_tensor(boxes_array, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        
        # Convert to tensor
        img = F.to_tensor(img)

        # TEXTURE/COLOR AUGMENTATIONS (Do not modify bounding boxes)
        if self.split == "train":
            if random.random() < 0.5:
                img = F.adjust_brightness(img, brightness_factor=random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                img = F.adjust_contrast(img, contrast_factor=random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                # NEW: Gaussian Blur to prevent texture memorization
                img = F.gaussian_blur(img, kernel_size=[3, 3])

        return img, target

    def __len__(self):
        return len(self.images)

def collate_fn(batch):
    return tuple(zip(*batch))

# =====================================================
# EVALUATION HELPER FUNCTIONS
# =====================================================

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)

def evaluate_map(model, data_loader, device, num_classes, conf_thresh, iou_thresh):
    model.eval()
    class_scores = {c: [] for c in range(1, num_classes)}
    class_matches = {c: [] for c in range(1, num_classes)}
    total_gt = {c: 0 for c in range(1, num_classes)}

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for b in range(len(images)):
                pred_boxes = outputs[b]["boxes"].cpu().numpy()
                pred_scores = outputs[b]["scores"].cpu().numpy()
                pred_labels = outputs[b]["labels"].cpu().numpy()

                gt_boxes = targets[b]["boxes"].cpu().numpy()
                gt_labels = targets[b]["labels"].cpu().numpy()

                for c in range(1, num_classes):
                    total_gt[c] += np.sum(gt_labels == c)

                keep = pred_scores >= conf_thresh
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                matched_gt = []
                for i in range(len(pred_boxes)):
                    cls = pred_labels[i]
                    best_iou = 0
                    best_gt_idx = -1

                    for j in range(len(gt_boxes)):
                        iou = compute_iou(pred_boxes[i], gt_boxes[j])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                    if best_iou >= iou_thresh and best_gt_idx not in matched_gt and gt_labels[best_gt_idx] == cls:
                        class_matches[cls].append(1)
                        matched_gt.append(best_gt_idx)
                    else:
                        class_matches[cls].append(0)

                    class_scores[cls].append(pred_scores[i])

    aps = []
    for c in range(1, num_classes):
        if total_gt[c] == 0:
            continue
        scores = np.array(class_scores[c])
        matches = np.array(class_matches[c])
        if len(scores) == 0:
            aps.append(0.0)
            continue

        sorted_idx = np.argsort(-scores)
        matches = matches[sorted_idx]

        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (total_gt[c] + 1e-6)

        ap = 0
        for i in range(1, len(precision)):
            ap += precision[i] * (recall[i] - recall[i-1])
        aps.append(ap)

    return np.mean(aps) if len(aps) > 0 else 0.0

# =====================================================
# DATA LOADERS & MODEL
# =====================================================

train_dataset = GalaxyDataset(DATA_ROOT, "train")
val_dataset   = GalaxyDataset(DATA_ROOT, "val")

# Safe to use num_workers > 0 now that Slurm cores are properly allocated
num_workers = 4 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)

# APPLIED WEIGHT DECAY
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=SCHEDULER_PATIENCE)

# =====================================================
# TRAINING & VALIDATION
# =====================================================

print("\n[TRAIN] Eğitim başlıyor...\n", flush=True)

train_loss_history = []
val_loss_history = []
val_map_history = []  

best_mAP = 0.0  
patience_counter = 0  

for epoch in range(EPOCHS):

    # --- TRAINING PHASE ---
    model.train()
    epoch_train_loss = 0
    optimizer.zero_grad() 

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values()) / ACCUMULATION_STEPS
        losses.backward()
        
        epoch_train_loss += (losses.item() * ACCUMULATION_STEPS)

        if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # --- VALIDATION PHASE (Loss) ---
    epoch_val_loss = 0
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images = [img.to(DEVICE) for img in val_images]
            val_targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in val_targets]

            val_loss_dict = model(val_images, val_targets)
            val_losses = sum(loss for loss in val_loss_dict.values())
            epoch_val_loss += val_losses.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    # --- VALIDATION PHASE (mAP) ---
    current_mAP = evaluate_map(model, val_loader, DEVICE, NUM_CLASSES, CONF_THRESHOLD, IOU_THRESHOLD)
    val_map_history.append(current_mAP)
    
    scheduler.step(current_mAP)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"[EPOCH {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mAP@0.5: {current_mAP:.4f} | LR: {current_lr:.6f}", flush=True)

    # --- SAVE LOGIC & EARLY STOPPING ---
    if current_mAP > best_mAP:
        best_mAP = current_mAP
        torch.save(model.state_dict(), "best_galaxy_model.pth")
        print(f" --> Yeni en iyi model kaydedildi! (mAP: {best_mAP:.4f})", flush=True)
        patience_counter = 0  
    else:
        patience_counter += 1
        print(f" [Early Stopping] Patience: {patience_counter}/{PATIENCE} (Best mAP: {best_mAP:.4f})", flush=True)

    if patience_counter >= PATIENCE:
        print(f"\n[Early Stopping] Triggered at epoch {epoch+1}. mAP did not improve for {PATIENCE} consecutive epochs.", flush=True)
        break

torch.save(model.state_dict(), "last_galaxy_model.pth")

# =====================================================
# PLOTTING DIAGNOSTICS
# =====================================================

# 1. Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label="Training Loss", color="#1f77b4", linewidth=2)
plt.plot(val_loss_history, label="Validation Loss", color="#d62728", linewidth=2, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Convergence: Training vs. Validation Loss")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.savefig("loss_graph_combined.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. mAP Graph
plt.figure(figsize=(10, 6))
plt.plot(val_map_history, label="Validation mAP@0.5", color="#2ca02c", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Model Accuracy: Mean Average Precision over Epochs")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.savefig("map_graph.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n[TRAIN] Eğitim tamamlandı. Grafikler kaydedildi.\n", flush=True)

# =====================================================
# FINAL EVALUATION (Confusion Matrix & Visuals)
# =====================================================

print("[EVAL] Son Değerlendirme başlıyor...\n", flush=True)

model.load_state_dict(torch.load("best_galaxy_model.pth", weights_only=True))
model.eval()

all_gt_labels = []
all_pred_labels = []

os.makedirs("predictions", exist_ok=True)

with torch.no_grad():
    for idx, (images, targets) in enumerate(val_loader):
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        pred_boxes = outputs[0]["boxes"].cpu().numpy()
        pred_scores = outputs[0]["scores"].cpu().numpy()
        pred_labels = outputs[0]["labels"].cpu().numpy()

        gt_boxes = targets[0]["boxes"].cpu().numpy()
        gt_labels = targets[0]["labels"].cpu().numpy()

        keep = pred_scores >= CONF_THRESHOLD
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        matched_gt = []
        for i in range(len(pred_boxes)):
            cls = pred_labels[i]
            best_iou = 0
            best_gt_idx = -1
            for j in range(len(gt_boxes)):
                iou = compute_iou(pred_boxes[i], gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= IOU_THRESHOLD and best_gt_idx not in matched_gt:
                matched_gt.append(best_gt_idx)
                all_gt_labels.append(gt_labels[best_gt_idx])
                all_pred_labels.append(cls)

        img = images[0].permute(1,2,0).cpu().numpy()
        img = (img*255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = map(int, box)
            text = f"{CLASS_NAMES[label]} {score:.2f}"
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, text, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imwrite(f"predictions/pred_{idx}.jpg", img)

if len(all_gt_labels) > 0:
    cm = confusion_matrix(all_gt_labels, all_pred_labels, labels=[1,2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["MG","CG"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    with open("metrics_summary.txt", "w") as f:
        f.write(f"Best Overall mAP@0.5: {best_mAP:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

print("\n[PIPELINE TAMAMLANDI]", flush=True)
