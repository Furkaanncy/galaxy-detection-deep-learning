import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay

# =========================
# PATHS
# =========================

DATA_ROOT = "dataset"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

TRAIN_IMG = os.path.join(DATA_ROOT,"train","images")
TRAIN_LAB = os.path.join(DATA_ROOT,"train","labels")

VAL_IMG = os.path.join(DATA_ROOT,"val","images")
VAL_LAB = os.path.join(DATA_ROOT,"val","labels")

TEST_IMG = os.path.join(DATA_ROOT,"test","images")
TEST_LAB = os.path.join(DATA_ROOT,"test","labels")


# =========================
# BOZUK IMAGE TEMİZLEME
# =========================

def clean_images(folder):

    removed=0

    for f in os.listdir(folder):

        path=os.path.join(folder,f)

        img=cv2.imread(path)

        if img is None:

            print("Bozuk görüntü silindi:",path)

            os.remove(path)

            removed+=1

    print("Toplam silinen:",removed)


clean_images(TRAIN_IMG)
clean_images(VAL_IMG)
clean_images(TEST_IMG)


# =========================
# DATASET
# =========================

class GalaxyDataset(Dataset):

    def __init__(self,img_dir,label_dir):

        self.img_dir=img_dir
        self.label_dir=label_dir

        self.images=[f for f in os.listdir(img_dir)
                     if f.endswith((".jpg",".png",".jpeg"))]

        self.images.sort()

        print(img_dir,"görüntü sayısı:",len(self.images))


    def __len__(self):

        return len(self.images)


    def __getitem__(self,idx):

        name=self.images[idx]

        img_path=os.path.join(self.img_dir,name)

        label_path=os.path.join(self.label_dir,
                                name.replace(".jpg",".txt")
                                .replace(".png",".txt")
                                .replace(".jpeg",".txt"))

        img=cv2.imread(img_path)

        if img is None:

            print("Görüntü okunamadı:",img_path)

            raise RuntimeError("image error")

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        h,w,_=img.shape

        boxes=[]
        labels=[]

        if os.path.exists(label_path):

            with open(label_path) as f:

                for line in f.readlines():

                    c,x,y,bw,bh=map(float,line.split())

                    x1=(x-bw/2)*w
                    y1=(y-bh/2)*h
                    x2=(x+bw/2)*w
                    y2=(y+bh/2)*h

                    boxes.append([x1,y1,x2,y2])

                    labels.append(int(c)+1)

        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.int64)

        img=torch.from_numpy(img).permute(2,0,1).float()/255

        target={}
        target["boxes"]=boxes
        target["labels"]=labels

        return img,target


def collate(batch):

    imgs=[b[0] for b in batch]
    targets=[b[1] for b in batch]

    return imgs,targets


# =========================
# DATA LOADERS
# =========================

train_dataset=GalaxyDataset(TRAIN_IMG,TRAIN_LAB)
val_dataset=GalaxyDataset(VAL_IMG,VAL_LAB)
test_dataset=GalaxyDataset(TEST_IMG,TEST_LAB)

train_loader=DataLoader(train_dataset,batch_size=2,shuffle=True,collate_fn=collate)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,collate_fn=collate)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=collate)


# =========================
# MODEL-
# ========================
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import SSD300_VGG16_Weights

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

num_classes=3

model.head.classification_head.num_classes=num_classes

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=model.to(device)


# =========================
# OPTIMIZER
# =========================

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)


# =========================
# TRAIN
# =========================

epochs=3
loss_history=[]

print("Training başladı")

for epoch in range(epochs):

    model.train()

    total_loss=0

    for imgs,targets in tqdm(train_loader):

        imgs=[img.to(device) for img in imgs]

        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict=model(imgs,targets)

        losses=sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss+=losses.item()

    epoch_loss=total_loss/len(train_loader)

    loss_history.append(epoch_loss)

    print("Epoch",epoch+1,"Loss:",epoch_loss)


# =========================
# LOSS GRAPH
# =========================

plt.figure()

plt.plot(loss_history)

plt.title("Training Loss")

plt.savefig(os.path.join(RESULT_DIR,"loss.png"))


# =========================
# EVALUATION
# =========================

model.eval()

y_true=[]
y_pred=[]
scores_all=[]
true_all=[]

for imgs,targets in tqdm(test_loader):
    
    imgs=[img.to(device) for img in imgs]

    preds=model(imgs)

    pred=preds[0]

    labels=pred["labels"].detach().cpu().numpy()
    scores=pred["scores"].detach().cpu().numpy()
    boxes=pred["boxes"].detach().cpu().numpy()

    gt_labels=targets[0]["labels"].numpy()

    for l in gt_labels:

        y_true.append(l)

    for l in labels[:len(gt_labels)]:

        y_pred.append(l)

    scores_all.extend(scores[:len(gt_labels)])
    true_all.extend(gt_labels)
true_all = np.array(true_all)

true_all = np.where(true_all == 2, 1, 0)

precision, recall, _ = precision_recall_curve(true_all, scores_all)

# =========================
# CONFUSION MATRIX
# =========================

cm=confusion_matrix(y_true,y_pred)

disp=ConfusionMatrixDisplay(cm,display_labels=["MG","CG"])

disp.plot()

plt.savefig(os.path.join(RESULT_DIR,"confusion_matrix.png"))


# =========================
# PR CURVE
# =========================

precision,recall,_=precision_recall_curve(true_all,scores_all)

plt.figure()

plt.plot(recall,precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.savefig(os.path.join(RESULT_DIR,"PR_curve.png"))


# =========================
# PREDICTIONS
# =========================

for i in range(min(5,len(test_dataset))):

    img,_=test_dataset[i]

    input_img=img.unsqueeze(0).to(device)

    pred=model(input_img)[0]

    img_np=img.permute(1,2,0).numpy()

    for box in pred["boxes"][:5]:

        x1,y1,x2,y2=box.cpu().numpy()

        cv2.rectangle(img_np,(int(x1),int(y1)),
                      (int(x2),int(y2)),(255,0,0),2)

    plt.imshow(img_np)
    plt.axis("off")

    plt.savefig(os.path.join(RESULT_DIR,f"prediction_{i}.png"))

print("Pipeline tamamlandı")