import os
import logging
import csv
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from data.mustard.image_transform import image_transforms

logger = logging.getLogger(__name__)

class Mustard(Dataset):
    def __init__(self, mode, limit=None):
        self.data = self.load_data(mode, limit)
        self.image_ids = list(self.data.keys())
    
    def load_data(self, mode, limit):
        cnt = 0
        data_set = dict()
        if mode in ['train']:
            df = pd.read_csv('data/mustard/'+mode+'_mustard.csv')
            for index, row in df.iterrows():
                if limit != None and cnt >= limit:
                    break

                image = row['image']
                sentence = row['sentence']
                label = row['label']
                image_path = 'data/mustard/img/'+image+'.jpg'

                if os.path.isfile(image_path):
                    data_set[image]={'text':sentence, 'label': label, 'image_path': image_path}
                    cnt += 1
        
        if mode in ['test','val']:
            df = pd.read_csv('data/mustard/'+mode+'_mustard.csv')
            for index, row in df.iterrows():
                image = row['image']
                sentence = row['sentence']
                label = row['label']
                image_path = 'data/mustard/img/'+image+'.jpg'

                if os.path.isfile(image_path):
                    data_set[image]={'text':sentence, 'label': label, 'image_path': image_path}
                    cnt += 1

        return data_set


    def __getitem__(self, index):
        id = self.image_ids[index]
        text = self.data[id]['text']
        video = image_transforms(id)
        label = self.data[id]['label']
        return text, video, label, id

    def __len__(self):
        return len(self.image_ids)
    
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text = []
        videos = []
        labels = []
        id_list = []
        for instance in batch_data:
            text.append(instance[0])
            videos.append(instance[1])
            # videos.extend(instance[1])
            labels.append(instance[2])
            id_list.append(instance[3])

        # Convert lists to tensors
        image_tensor = torch.stack(videos, dim=0)

        return text, image_tensor, labels, id_list