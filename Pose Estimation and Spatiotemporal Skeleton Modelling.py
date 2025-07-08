import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# ------------------------- Pose-GCN -------------------------
class PoseGCN(nn.Module):
    def __init__(self, in_channels=2, out_channels=64, num_joints=17):
        super(PoseGCN, self).__init__()
        self.conv1 = nn.Conv1d(num_joints * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, T, J, C]
        B, T, J, C = x.size()
        x = x.view(B, T, J * C).permute(0, 2, 1)  # [B, JC, T]
        x = self.conv1(x)  # [B, F, T]
        return x

# ------------------------- Transformer Lite -------------------------
class TransformerLite(nn.Module):
    def __init__(self, input_dim, n_heads=4, ff_dim=128):
        super(TransformerLite, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [T, B, F]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# ------------------------- Multi-Head Attention -------------------------
class MultiHeadJointFrameAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)

    def forward(self, x):
        # x: [T, B, F]
        attn_output, _ = self.attn(x, x, x)
        return attn_output

# ------------------------- Behavioral Zone Masking -------------------------
class BehavioralZoneMasking(nn.Module):
    def __init__(self, zones=['exit', 'stairwell'], joint_indices=[15, 16]):
        super().__init__()
        self.zone_map = {zone: joint_indices for zone in zones}

    def forward(self, x, zone='exit'):
        # x: [B, T, J, C]
        zone_joints = self.zone_map.get(zone, [])
        mask = torch.zeros_like(x)
        for joint_idx in zone_joints:
            mask[:, :, joint_idx, :] = x[:, :, joint_idx, :]
        return mask

# ------------------------- Full Action Recognition Model -------------------------
class ActionRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_gcn = PoseGCN()
        self.zone_mask = BehavioralZoneMasking()
        self.transformer = TransformerLite(input_dim=64)
        self.joint_attention = MultiHeadJointFrameAttention(dim=64)
        self.classifier = nn.Linear(64, 10)  # 10 classes (adjust as needed)

    def forward(self, x, zone='exit'):
        # x: [B, T, J, C]
        zone_data = self.zone_mask(x, zone)  # focus on zone
        gcn_out = self.pose_gcn(zone_data)  # [B, F, T]
        gcn_out = gcn_out.permute(2, 0, 1)  # [T, B, F]
        trans_out = self.transformer(gcn_out)
        attn_out = self.joint_attention(trans_out)  # [T, B, F]
        pooled = attn_out.mean(dim=0)  # [B, F]
        return self.classifier(pooled)
# 📂 Set your input image folder
input_folder = ''

# 🎯 Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# 🔁 Process all images in the folder
for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Couldn't read {filename}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run pose estimation
        results = pose.process(image_rgb)

        # Create outputs
        skeleton_blank = np.ones_like(image_rgb) * 255  # White background
        skeleton_overlay = image_rgb.copy()

        print(f"\n📄 Processing: {filename}")

        if results.pose_landmarks:
            # Draw landmarks on skeleton overlay
            mp_drawing.draw_landmarks(
                skeleton_overlay,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Draw on white background
            mp_drawing.draw_landmarks(
                skeleton_blank,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
            )
        else:
            print("⚠️ No person detected.")

        # Plot side-by-side
        plt.figure(figsize=(18, 6))
        plt.suptitle(f"Pose Estimation - {filename}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(skeleton_overlay)
        plt.title("Skeleton Overlay")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(skeleton_blank.astype(np.uint8))
        plt.title("Skeleton on White")
        plt.axis("off")

        plt.tight_layout()
        plt.show()