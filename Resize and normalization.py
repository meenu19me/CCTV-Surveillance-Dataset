import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_and_normalize(image, size=(224, 224)):
    """
    Resize the image and normalize pixel values to [0, 1].

    Args:
        image (numpy array): Original BGR image from OpenCV.
        size (tuple): Target size (width, height) for resizing.

    Returns:
        numpy array: Normalized RGB image.
    """
    resized = cv2.resize(image, size)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    return normalized

def process_images(main_folder, output_folder):
    """
    Process all images by resizing and normalizing, display and save folder-wise.

    Args:
        main_folder (str): Path to the input image folder.
        output_folder (str): Path where processed images will be saved.
    """
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                original = cv2.imread(image_path)

                if original is None:
                    continue

                processed = resize_and_normalize(original)

                # Convert back to uint8 for saving
                processed_uint8 = (processed * 255).astype(np.uint8)
                processed_bgr = cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)

                # Recreate folder structure
                rel_path = os.path.relpath(root, main_folder)
                save_dir = os.path.join(output_folder, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, processed_bgr)

                # Display original and processed image side-by-side
                plt.figure(figsize=(8, 4))

                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Resized & Normalized")
                plt.imshow(processed)
                plt.axis('off')

                plt.tight_layout()
                plt.show()

# =============================
# ✅ Example usage:
# =============================
input_folder = ""
output_folder = ""

process_images(input_folder, output_folder)