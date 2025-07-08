import cv2
import os
from google.colab.patches import cv2_imshow

def display_and_save_sampled_frames(main_folder, output_folder, sample_fps=10):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith('.avi'):
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, main_folder)
                save_subfolder = os.path.join(output_folder, relative_path, os.path.splitext(file)[0])
                os.makedirs(save_subfolder, exist_ok=True)

                print(f"\nProcessing: {video_path}")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open {video_path}")
                    continue

                # Get the video frame rate
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0:
                    print(f"Unable to get FPS for {video_path}, skipping.")
                    cap.release()
                    continue

                # Calculate sampling interval
                frame_interval = int(video_fps / sample_fps)
                if frame_interval <= 0:
                    frame_interval = 1

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                saved_frame_idx = 0

                while frame_count < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    #Display original frame
                    cv2_imshow(frame)
                    print(f"🖼️ Displayed Frame at position {frame_count} (Original size)")

                    # ✅ Save original frame
                    output_path = os.path.join(save_subfolder, f'frame_{saved_frame_idx:03}.jpg')
                    cv2.imwrite(output_path, frame)
                    print(f"💾 Saved to: {output_path}")

                    saved_frame_idx += 1
                    frame_count += frame_interval

                cap.release()

# ✅ Example usage
main_folder_path = ''
output_folder_path = ''

# Sample at 10 FPS (can be changed as needed)
display_and_save_sampled_frames(main_folder_path, output_folder_path, sample_fps=10)