import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MustardVideoText(Dataset):
    def __init__(self, args, device, videos_path, text_path, labels_path, ids_path):
        self.args = args
        self.device = device

        self.videos = torch.load(videos_path)
        self.text = torch.load(text_path)
        self.labels = torch.load(labels_path)
        # self.vid_id = torch.load(ids_path)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        video = self.videos[index]
        text = self.text[index]
        label = self.labels[index]
        return video, text, label
    
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return {}

        videos = []
        text = []
        labels = []

        for instance in batch_data:
            videos.append(instance[0])
            text.append(instance[1])
            labels.append(instance[2])

        videos = torch.stack(videos, dim=0)

        return videos, text, labels


class MustardText(Dataset):
    def __init__(self, args, device, text_path, labels_path):
        super().__init__()
        self.args = args
        self.device = device

        self.text = torch.load(text_path)
        self.labels = torch.load(labels_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text = self.text[index]
        label = self.labels[index]

        return text, label
    
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
        if batch_size == 0:
            return {}

        text = []
        labels = []

        for instance in batch_data:
            text.append(instance[0])
            labels.append(instance[1])

        return text, labels
