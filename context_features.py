import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DeepLabV3 for background segmentation
deeplab = deeplabv3_resnet101(pretrained=True).eval().to(device)

def segment_subject(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = deeplab(input_tensor)['out'][0]
    mask = output.argmax(0).cpu().numpy()
    # Mask for the subject: 0 is background, so the subject is everything else
    subject_mask = mask != 0  # Everything except background is the subject
    return subject_mask

# Load MiDaS model
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device)
midas.eval()

# Load MiDaS transform
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transform.dpt_transform
else:
    transform = midas_transform.small_transform

def compute_depth(image):
    """
    Computes depth map using MiDaS.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = np.array(image)

    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    input_batch = transform(image).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

# Function to process all videos in the dataset
def process_dataset(dataset_path, output_pkl):
    video_features = {}

    for root, dirs, files in os.walk(dataset_path):
        for video_file in files:
            if video_file.endswith(('.mp4', '.avi', '.mov')):  # Filter video files
                video_path = os.path.join(root, video_file)
                video_name = os.path.splitext(video_file)[0]  # Use the video filename without extension as the key
                print(f"Processing video: {video_name}")

                cap = cv2.VideoCapture(video_path)

                # Store features for the current video
                video_frames = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Get the subject mask (everything except the background)
                    subject_mask = segment_subject(pil_image)

                    # Apply the mask to keep only the background
                    background_masked_image = frame_rgb.copy()
                    background_masked_image[subject_mask] = [0, 0, 0]  # Set subject to black

                    # Compute the depth map
                    depth_map = compute_depth(pil_image)

                    # Append the features (subject-masked image and depth map)
                    video_frames.append({
                        "frame": frame_rgb,
                        "background_masked": background_masked_image,
                        "depth_map": depth_map
                    })

                cap.release()

                # Store the processed frames for the current video
                video_features[video_name] = video_frames

    # Save the features to a pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(video_features, f)
    print(f"Features saved to {output_pkl}")

# Set paths
dataset_path = ".\\dataset\\MOSEI\\Raw"
output_pkl = ".\\dataset\\MOSEI\\Processed\\context_features.pkl"

# Process the dataset
process_dataset(dataset_path, output_pkl)