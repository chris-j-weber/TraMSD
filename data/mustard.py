import os
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict 
from torch.utils.data import Dataset
from config import Config
from utils.video_capture import VideoCapture

class Mustard(Dataset):
    
    def __init__(self, config: Config, mode = 'train', img_transforms = None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        # self.num_frames = config.num_frames
        self.mode = mode

        db_file = 'data/mustard++_text.csv'
        self.db = pd.read_csv(db_file)
        # self.db = self.db.sample(frac=0.1)
        
        # split dataset train, val and test
        self.train_videos = self.db['scene'].unique()
        # video2caption
        # all_video_pairs

    def __len__(self):
        return len(self.train_videos)

    def __getitem__(self, index):
        video_path, caption, scene = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)
        # video_sample_type ???

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'scene': scene,
            'video': imgs,
            'text': caption
        }