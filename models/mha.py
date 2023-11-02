import math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = args.embed_dim
        self.num_heads = args.num_mha
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, text_embeds, image_embeds):
        num_texts, _ = text_embeds.shape

        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        
        # num_heads x head_dim x num_texts
        q = q.permute(1, 2, 0)

        num_vids, num_frames, _ = image_embeds.shape
        
        # num_vids x num_frames x embed_dim
        k = self.k_proj(image_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0, 2, 1, 3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(image_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0, 2, 3, 1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights

        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0, 3, 1, 2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)

        return o