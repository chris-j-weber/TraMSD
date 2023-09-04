import torch
import torch.nn as nn
from config import Config
from models.transformer import Transformer

class CLIP(nn.Module):
    def __init__(self, config: Config):
        super(CLIP, self).__init__()
        self.config = config
        
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

        config.pooling = 'transformer'
        self.pool_frames = Transformer(config)

    def forward(self, data,  return_all_frames=False):
        batch_size = data['video'].shape[0]
        #num_frames = data['video'].shape[1]
        text_data = data['text']
        video_data = data['video']
        #video_data = video_data.reshape(batch_size*num_frames, 3, 224, 224)
        video_data = video_data.reshape(-1, 3, self.config.input_resolution, self.config.input_resolution)

        text_features = self.clip.get_text_features(**text_data)
        video_features = self.clip.get_image_features(video_data)

        video_features = video_features.reshape(batch_size, self.config.num_frames, -1)
        video_features_pooled = self.pool_frames(text_features, video_features)

        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled