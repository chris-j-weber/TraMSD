import cv2
import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

def capture_frames(video_id, video_path, num_frames):
    cam = cv2.VideoCapture(video_path)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    processed_frames = []

    samples = min(num_frames, total_frames)
    random_frame_idxs = random.sample(range(total_frames), samples)

    if not cam.isOpened():
        print('Error opening video stream or file')
    else:
        for frame_idx in sorted(random_frame_idxs):
            cam.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cam.read()
            if not ret:
                print(f'frame {frame_idx} could not be read')
                continue

            output_filename = f'data/mustard/frames/{video_id}_{frame_idx}.jpg'
            cv2.imwrite(output_filename, frame)

    cam.release()
    cv2.destroyAllWindows()
    
    return None

def image_transforms(video_id):
    frames = []
    processed_frames = []
    folder_path = 'data/mustard/frames'
    trans = transforms.Compose([transforms.ToTensor()])

    for filename in os.listdir(folder_path):
        if filename.startswith(video_id):
            output_filename = os.path.join(folder_path, filename)
            frame = Image.open(output_filename).convert('RGB')
            frames.append(frame)

    for fram in frames:
        resized_frame = F.resize(fram, size=(224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        transformed_frame = trans(resized_frame)
        processed_frames.append(transformed_frame)

    video = torch.stack(processed_frames, dim=0).float()
    video = video.permute(0, 2, 3, 1)

    return video