import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef)
import numpy as np
import os
import glob

# Model Definitions
class DenseGRUBackbone(nn.Module):
    def __init__(self, hidden_dim=128):
        super(DenseGRUBackbone, self).__init__()
        self.densenet = densenet121(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.gru = nn.GRU(input_size=1024, hidden_size=hidden_dim, batch_first=True)
    def forward(self, x):
        B, T, C, H, W = x.shape
        features = []
        for t in range(T):
            xt = self.densenet(x[:, t])
            xt = self.pool(xt).squeeze(-1).squeeze(-1)
            features.append(xt)
        features = torch.stack(features, dim=1)
        gru_out, _ = self.gru(features)
        return gru_out[:, -1]

class MLPFusionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=10):
        super(MLPFusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class CombinedDenseGRU_MLP(nn.Module):
    def __init__(self, gru_hidden_dim=128, mlp_hidden_dim=64, num_classes=10):
        super(CombinedDenseGRU_MLP, self).__init__()
        self.backbone = DenseGRUBackbone(hidden_dim=gru_hidden_dim)
        self.classifier = MLPFusionModel(input_dim=gru_hidden_dim, hidden_dim=mlp_hidden_dim, num_classes=num_classes)
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_video_frames(video_folder, max_frames=5):
    frame_paths = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))
    frames = []
    for path in frame_paths[:max_frames]:
        image = Image.open(path).convert("RGB")
        frames.append(transform(image))
    while len(frames) < max_frames:
        frames.append(frames[-1])
    return torch.stack(frames)  # [T, C, H, W]

def compute_all_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)
    epsilon = 1e-7
    specificity = np.mean(TN / (TN + FP + epsilon))
    npv = np.mean(TN / (TN + FN + epsilon))
    fpr = np.mean(FP / (FP + TN + epsilon))
    fnr = np.mean(FN / (FN + TP + epsilon))
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'specificity': specificity, 'mcc': mcc, 'npv': npv, 'fpr': fpr, 'fnr': fnr}

def evaluate_model(model, dataset_root, num_classes=10):
    y_true, y_pred = [], []
    class_folders = sorted([f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))])
    label_mapping = {name: idx for idx, name in enumerate(class_folders)}
    for class_folder in class_folders:
        class_path = os.path.join(dataset_root, class_folder)
        label = label_mapping.get(class_folder, None)
        if label is None:
            continue
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            if not os.path.isdir(video_path):
                continue
            try:
                video_tensor = load_video_frames(video_path).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(video_tensor)
                    pred = torch.argmax(output, dim=1).item()
                y_pred.append(pred)
                y_true.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {video_path}: {e}")
    return compute_all_metrics(y_true, y_pred)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedDenseGRU_MLP(num_classes=10).to(device)
    dataset_root = ""
    results = evaluate_model(model, dataset_root, num_classes=10)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")