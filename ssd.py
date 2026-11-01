import os
import cv2
import torch
import torchvision
import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import warnings
# Ignore the harmless torchmetrics warning about having more than 100 predictions per image
warnings.filterwarnings("ignore", message="Encountered more than 100 detections in a single image")

# =========================
# PATHS
# =========================

DATA_ROOT = "/arf/scratch/bulas/YL_tez/280226/ssd/dataset"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)
TRAIN_IMG = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LAB = os.path.join(DATA_ROOT, "train", "labels")

VAL_IMG = os.path.join(DATA_ROOT, "val", "images")
VAL_LAB = os.path.join(DATA_ROOT, "val", "labels")

TEST_IMG = os.path.join(DATA_ROOT, "test", "images")
TEST_LAB = os.path.join(DATA_ROOT, "test", "labels")


# =========================
# BOZUK IMAGE TEMİZLEME
# =========================

def clean_images(folder):
    removed = 0
    if not os.path.exists(folder):
        print(f"Klasör bulunamadı: {folder}")
        return

    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        img = cv2.imread(path)
        if img is None:
            print("Bozuk görüntü silindi:", path)
            os.remove(path)
            removed += 1
    print("Toplam silinen:", removed)

clean_images(TRAIN_IMG)
clean_images(VAL_IMG)
clean_images(TEST_IMG)


# =========================
# DATASET W/ ADVANCED AUGMENTATION
# =========================

class GalaxyDataset(Dataset):
    def __init__(self, img_dir, label_dir, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augment = augment
        self.images = [f for f in os.listdir(img_dir)
                       if f.endswith((".jpg", ".png", ".jpeg"))]
        self.images.sort()
        print(img_dir, "görüntü sayısı:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)
        label_path = os.path.join(self.label_dir,
                                  name.replace(".jpg", ".txt")
                                  .replace(".png", ".txt")
                                  .replace(".jpeg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            print("Görüntü okunamadı:", img_path)
            raise RuntimeError("image error")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    c, x, y, bw, bh = map(float, line.split())
                    x1 = (x - bw / 2) * w
                    y1 = (y - bh / 2) * h
                    x2 = (x + bw / 2) * w
                    y2 = (y + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c) + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # --- Advanced Augmentation Logic ---
        if self.augment:
            # 1. Random Brightness & Contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2) 
                beta = random.uniform(-20, 20)   
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            # 2. Horizontal Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                if len(boxes) > 0:
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]] 
            
            # 3. Vertical Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)
                if len(boxes) > 0:
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                    
            # 4. Random 90-Degree Clockwise Rotation
            if random.random() > 0.5:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                if len(boxes) > 0:
                    new_boxes = boxes.clone()
                    new_boxes[:, 0] = h - boxes[:, 3] 
                    new_boxes[:, 1] = boxes[:, 0]     
                    new_boxes[:, 2] = h - boxes[:, 1] 
                    new_boxes[:, 3] = boxes[:, 2]     
                    boxes = new_boxes
                h, w = img.shape[:2] 
        # -----------------------------------

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

def collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return imgs, targets


# =========================
# DATA LOADERS
# =========================

train_dataset = GalaxyDataset(TRAIN_IMG, TRAIN_LAB, augment=True)
val_dataset = GalaxyDataset(VAL_IMG, VAL_LAB, augment=False)
test_dataset = GalaxyDataset(TEST_IMG, TEST_LAB, augment=False)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)


# =========================
# MODEL
# ========================

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
num_classes = 3 # Background (0) + MG (1) + CG (2)
model.head.classification_head.num_classes = num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# =========================
# OPTIMIZER & SCHEDULER
# =========================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Monitors MAX mAP instead of MIN loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=3,
    verbose=True
)


# =========================
# TRAIN, VALIDATION & MAP EVALUATION W/ EARLY STOPPING
# =========================

epochs = 50
patience = 7  

train_loss_history = []
val_loss_history = []
val_map_history = []  

best_map_50 = 0.0 
epochs_no_improve = 0

print("Training başladı")

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    total_train_loss = 0

    for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()

        total_train_loss += losses.item()

    epoch_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(epoch_train_loss)

    # --- VALIDATION LOSS PHASE ---
    model.train() 
    total_val_loss = 0
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Loss]"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
            
    epoch_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(epoch_val_loss)
    
    # --- VALIDATION mAP PHASE ---
    model.eval()
    epoch_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    with torch.no_grad():
        for imgs, targets in val_loader: 
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)
            
            preds_cpu = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()} for p in preds]
            targets_cpu = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
            
            epoch_metric.update(preds_cpu, targets_cpu)
            
    # Compute the mAP for this epoch
    mAP_results = epoch_metric.compute()
    current_map_50 = mAP_results['map_50'].item()
    val_map_history.append(current_map_50)
    
    # Step the scheduler based on mAP_50
    scheduler.step(current_map_50)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val mAP@50: {current_map_50:.4f} | LR: {current_lr:.2e}")

    # --- EARLY STOPPING CHECK ---
    if current_map_50 > best_map_50:
        best_map_50 = current_map_50
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model.pth"))
        print(f"  [+] mAP improved to {best_map_50:.4f}. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"  [-] No improvement in mAP. Patience: {epochs_no_improve}/{patience}")
        
        if epochs_no_improve >= patience:
            print("  [!] Early stopping triggered based on plateauing mAP! Training halted.")
            break

print("Loading best model weights for evaluation...")
model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_model.pth"), weights_only=True))


# =========================
# GRAPHS (LOSS AND MAP)
# =========================

# 1. Loss Graph
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label="Training Loss", color="blue", linewidth=2)
plt.plot(val_loss_history, label="Validation Loss", color="orange", linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(RESULT_DIR, "loss.png"))
plt.close()

# 2. mAP Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(val_map_history) + 1), val_map_history, label="Validation mAP@50", color="green", linewidth=2, marker='o')
plt.title("Validation mAP@50 over Epochs")
plt.xlabel("Epochs")
plt.ylabel("mAP@50")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(RESULT_DIR, "map_history.png"))
plt.close()

print(f"Grafikler kaydedildi: {RESULT_DIR}/loss.png ve {RESULT_DIR}/map_history.png")


# =========================
# CUSTOM CONFUSION MATRIX CLASS
# =========================

class DetectionConfusionMatrix:
    def __init__(self, num_classes, iou_thresh=0.5):
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, preds, targets):
        for p, t in zip(preds, targets):
            pred_boxes = p["boxes"]
            pred_labels = p["labels"]
            pred_scores = p["scores"]
            
            gt_boxes = t["boxes"]
            gt_labels = t["labels"]

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
                
            if len(pred_boxes) == 0:
                for gl in gt_labels:
                    self.matrix[gl, 0] += 1
                continue
                
            if len(gt_boxes) == 0:
                for pl in pred_labels:
                    self.matrix[0, pl] += 1
                continue

            sort_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sort_idx]
            pred_labels = pred_labels[sort_idx]

            ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)

            for i, p_label in enumerate(pred_labels):
                best_iou, best_gt_idx = ious[i].max(dim=0) if ious.numel() > 0 else (0, -1)
                
                if best_iou >= self.iou_thresh and not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    true_label = gt_labels[best_gt_idx]
                    self.matrix[true_label, p_label] += 1
                else:
                    self.matrix[0, p_label] += 1

            for j, matched in enumerate(gt_matched):
                if not matched:
                    true_label = gt_labels[j]
                    self.matrix[true_label, 0] += 1

    def plot(self, save_path, class_names):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.matrix, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Object Detection Confusion Matrix (IoU > {self.iou_thresh})")
        plt.savefig(save_path)
        plt.close()
        print(f"Confusion matrix kaydedildi: {save_path}")


# =========================
# FINAL EVALUATION
# =========================

print("\nNihai Değerlendirme (Final Evaluation) başlıyor...")
model.eval()

metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
det_cm = DetectionConfusionMatrix(num_classes=3, iou_thresh=0.5)

# Threshold to filter out low-confidence "junk" boxes from the confusion matrix
CONFIDENCE_THRESHOLD = 0.50 

with torch.no_grad():
    for imgs, targets in tqdm(test_loader, desc="Evaluating Test Set"):
        imgs = [img.to(device) for img in imgs]
        preds = model(imgs)
        
        # 1. Prepare data for mAP (needs ALL predictions to build PR curve correctly)
        preds_cpu = [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(), "labels": p["labels"].cpu()} for p in preds]
        targets_cpu = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
        metric.update(preds_cpu, targets_cpu)

        # 2. Prepare data for Confusion Matrix (Filter out low confidence predictions)
        preds_cm = []
        for p in preds:
            keep_idx = p["scores"] > CONFIDENCE_THRESHOLD
            preds_cm.append({
                "boxes": p["boxes"][keep_idx].cpu(),
                "scores": p["scores"][keep_idx].cpu(),
                "labels": p["labels"][keep_idx].cpu()
            })
            
        det_cm.update(preds_cm, targets_cpu)

mAP_results = metric.compute()

print("\n--- Nihai Model Performans Sonuçları (mAP) ---")
readable_results = {k: round(v.item(), 4) for k, v in mAP_results.items() if v.numel() == 1}
pprint.pprint(readable_results)

det_cm.plot(
    save_path=os.path.join(RESULT_DIR, "confusion_matrix.png"), 
    class_names=["Background", "MG", "CG"]
)


# =========================
# PREDICTIONS
# =========================

print("\nÖrnek tahminler kaydediliyor...")
for i in range(min(5, len(test_dataset))):
    img, _ = test_dataset[i]
    input_img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(input_img)[0]

    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)

    for box in pred["boxes"][:5]:
        x1, y1, x2, y2 = box.cpu().numpy()
        cv2.rectangle(img_np, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 0), 2)

    plt.figure()
    plt.imshow(img_np)
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR, f"prediction_{i}.png"))
    plt.close()

print("Pipeline tamamlandı.")
