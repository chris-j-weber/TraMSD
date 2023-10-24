import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

def frame_loader(video_id):
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