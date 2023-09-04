import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class MHA(nn.Module):
    def __init__(self, config: Config):
        super(MHA, self).__init__()
        self.embedding_dimension = config.embedding_dimension
        self.num_heads = config.num_mha_heads
        assert self.embedding_dimension % self.num_heads == 0
        self.head_dim = self.embedding_dimension // self.num_heads

        self.q_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.k_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.v_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.o_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)

    def forward(self, text_embd, video_embd):
        num_text, _ = text_embd.shape
        q = self.q_projection(text_embd)
        q = q.reshape(num_text, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0)

        num_videos, num_frames, _ = video_embd.shape
        k = self.k_projection(video_embd)
        k = k.reshape(num_videos, num_frames, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = self.v_projection(video_embd)
        v = v.reshape(num_videos, num_frames, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)
        attention = v @ attention_weights
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_videos, num_text, self.emd_dim)

        o = self.o_projection(attention)
        return o
    
class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.emd_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MHA(config)

        self.linear_proj = nn.Linear(self.emd_dim, self.emd_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.emd_dim)
        self.layer_norm2 = nn.LayerNorm(self.emd_dim)
        self.layer_norm3 = nn.LayerNorm(self.emd_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out