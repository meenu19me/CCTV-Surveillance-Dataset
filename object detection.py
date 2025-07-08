import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# ------------------------- CBR, XQARepVGG, QSP, QCI Blocks -------------------------
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class XQARepVGG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XQARepVGG, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class QSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QSPBlock, self).__init__()
        self.cbr1 = CBR(in_channels, out_channels)
        self.repvgg_blocks = nn.Sequential(
            XQARepVGG(out_channels, out_channels),
            XQARepVGG(out_channels, out_channels),
            XQARepVGG(out_channels, out_channels)
        )
        self.concat_cbr = CBR(out_channels * 3, out_channels)

    def forward(self, x):
        x = self.cbr1(x)
        features = [blk(x) for blk in self.repvgg_blocks]
        x_concat = torch.cat(features, dim=1)
        return self.concat_cbr(x_concat)

class QCIBlock(QSPBlock):
    pass

# ------------------------- Backbone -------------------------
class YOLOBaselineBackbone(nn.Module):
    def __init__(self):
        super(YOLOBaselineBackbone, self).__init__()
        self.stage1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.stage2 = nn.Sequential(CBR(64, 256), CBR(256, 256))
        self.stage3 = nn.Sequential(CBR(256, 384), CBR(384, 384), CBR(384, 384))
        self.stage4 = nn.Sequential(CBR(384, 64), CBR(64, 64))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

# ------------------------- Neck -------------------------
class YOLONASNeck(nn.Module):
    def __init__(self):
        super(YOLONASNeck, self).__init__()
        self.up_stage6 = QSPBlock(64, 192)
        self.down_stage6 = QCIBlock(192, 128)
        self.up_stage8 = QSPBlock(64, 64)
        self.down_stage8 = QCIBlock(64, 64)

    def forward(self, x):
        u6 = self.up_stage6(x)
        d6 = self.down_stage6(u6)
        u8 = self.up_stage8(x)
        d8 = self.down_stage8(u8)
        return [d6, d8, x]

# ------------------------- Detection Head -------------------------
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(YOLOHead, self).__init__()
        self.detect = nn.Conv2d(in_channels, num_classes + 5, 1)

    def forward(self, x):
        return self.detect(x)

# ------------------------- Full Model -------------------------
class YOLO_NAS(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLO_NAS, self).__init__()
        self.backbone = YOLOBaselineBackbone()
        self.neck = YOLONASNeck()
        self.heads = nn.ModuleList([
            YOLOHead(128, num_classes),
            YOLOHead(64, num_classes),
            YOLOHead(64, num_classes)
        ])

    def forward(self, x):
        x = self.backbone(x)
        features = self.neck(x)
        outputs = [head(f) for head, f in zip(self.heads, features)]
        return outputs

# ------------------------- Helper Functions -------------------------
def load_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (416, 416))
    tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor, img_resized

def decode_predictions(pred, conf_thresh=0.4, default_box=True):
    pred = pred[0].permute(1, 2, 0).reshape(-1, 6)
    pred = pred[pred[:, 4] > conf_thresh]
    if pred.size(0) == 0 and default_box:
        x_center, y_center, width, height = 0.5, 0.5, 0.2, 0.2
        return np.array([[x_center, y_center, width, height]])
    return pred[:, :4].detach().cpu().numpy()

def draw_boxes(img, boxes, color=(255, 0, 0), label="Person"):
    for box in boxes:
        x, y, w, h = box
        x1 = int((x - w / 2) * img.shape[1])
        y1 = int((y - h / 2) * img.shape[0])
        x2 = int((x + w / 2) * img.shape[1])
        y2 = int((y + h / 2) * img.shape[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)
    return img

# ------------------------- YOLO_NAS Detection Function with Save -------------------------
def detect_person_YOLO_NAS(image_path, model, output_folder=None):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error reading image: {image_path}")
        return

    results = model(image_path)
    boxes = results[0].boxes
    person_indices = boxes.cls == 0
    if person_indices.sum() == 0:
        print("No person detected in:", image_path)
        return

    person_boxes = boxes.xyxy[person_indices].cpu().numpy()
    output_image = original_image.copy()

    for (x1, y1, x2, y2) in person_boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output_image, 'Person', (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save image
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, output_image)
        print(f"Saved output image to: {save_path}")

    # Display
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title("Detected: Person Only")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ------------------------- Process Folder -------------------------
def process_with_YOLO_NAS(folder_path, output_folder):
    model = YOLO('yolov5s.pt')
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_path = os.path.join(root, file)
                print(f"\n[YOLO_NAS] Processing: {image_path}")
                detect_person_YOLO_NAS(image_path, model, output_folder=output_folder)

# ------------------------- Main Run -------------------------
main_input_folder = ''
output_folder = ''

process_with_YOLO_NAS(main_input_folder, output_folder)