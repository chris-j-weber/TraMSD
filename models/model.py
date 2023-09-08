from transformers import CLIPModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions

class CLIP(nn.Module):
    def __init__(self, args):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)    
        self.text_linear =  nn.Sequential(
            nn.Linear(args.text_size, args.text_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
        self.image_linear =  nn.Sequential(
            nn.Linear(args.image_size, args.image_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
