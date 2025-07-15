import cv2
import os
import numpy as np
import shutil
from matplotlib import pyplot as plt

def is_camera_movement_significant(prev_img, curr_img, threshold=100):
    """
    Detect sudden camera movement using ORB feature matching.
    Returns True if movement is significant.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(prev_img, None)
    kp2, des2 = orb.detectAndCompute(curr_img, None)

    if des1 is None or des2 is None:
        return True  # Assume unstable if no features

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < threshold:
        return True
    return False

def drop_unstable_frames(input_folder, output_folder):
    """
    Traverse all subfolders, drop unstable frames, save/display stable ones.
    """
    for root, _, files in os.walk(input_folder):
        files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        prev_frame = None
        for file in files:
            input_path = os.path.join(root, file)
            curr_frame = cv2.imread(input_path)
            if curr_frame is None:
                continue

            gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Determine relative subfolder path
            rel_path = os.path.relpath(root, input_folder)
            output_subfolder = os.path.join(output_folder, rel_path)
            os.makedirs(output_subfolder, exist_ok=True)
            output_path = os.path.join(output_subfolder, file)

            if prev_frame is None:
                prev_frame = gray_curr
                cv2.imwrite(output_path, curr_frame)
                show_side_by_side(curr_frame, curr_frame)
                continue

            # Check for motion
            if not is_camera_movement_significant(prev_frame, gray_curr):
                cv2.imwrite(output_path, curr_frame)
                show_side_by_side(curr_frame, curr_frame)

            prev_frame = gray_curr

def show_side_by_side(input_img, output_img):
    """
    Display input and output images side-by-side using matplotlib.
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Frame")
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Saved (Stable)")
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==============================
# ✅ Example Usage
# ==============================
input_folder = ""
output_folder = ""

drop_unstable_frames(input_folder, output_folder)