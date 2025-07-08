import cv2
import os
import matplotlib.pyplot as plt

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0.2):
    """
    Apply Gaussian Blur to reduce camera noise.
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)

def process_images_with_blur(main_input_folder, main_output_folder):
    """
    Recursively apply Gaussian Blur to all images in main_input_folder and save to main_output_folder,
    maintaining the folder structure.

    Args:
        main_input_folder (str): Main folder containing subfolders of images.
        main_output_folder (str): Folder to save the blurred images in same structure.
    """
    for root, dirs, files in os.walk(main_input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                # Apply Gaussian Blur
                blurred = apply_gaussian_blur(image)

                # Reconstruct relative folder path
                relative_path = os.path.relpath(root, main_input_folder)
                output_dir = os.path.join(main_output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Save blurred image
                save_path = os.path.join(output_dir, file)
                cv2.imwrite(save_path, blurred)

                # Display original and blurred image side-by-side
                plt.figure(figsize=(8, 4))

                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Gaussian Blurred")
                plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.tight_layout()
                plt.show()

# ==============================
# ✅ Example Usage
# ==============================
input_folder = ""
output_folder = ""

process_images_with_blur(input_folder, output_folder)