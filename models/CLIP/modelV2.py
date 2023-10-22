import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer


class MHA(nn.Module):
    def __init__(self, args):
        super(MHA, self).__init__()
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
        q = q.permute(1,2,0)
        #q = q.mean(dim=1)
        #q = q.permute(1, 2, 0)

        num_vids, num_frames, _ = image_embeds.shape
        
        # num_vids x num_frames x embed_dim
        k = self.k_proj(image_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(image_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights

        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        # o = o.mean(dim=1)

        return o


class CLIPTransformer(nn.Module):
    def __init__(self, args):
        super(CLIPTransformer, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.config = BertConfig.from_pretrained("bert-base-uncased")
        # self.config.hidden_size = 512
        # self.config.num_attention_heads = args.num_mha
        
        self.embed_dim = args.embed_dim
        dropout = args.dropout_rate

        self.cross_attn = MHA(args)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.img_linear = nn.Linear(768, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(self.embed_dim, args.label_number)

        self.pooling = Transformer(args)

    def forward(self, inputs, labels):
        model_outputs = self.model(**inputs,output_attentions=True)

        txt_embed = model_outputs.text_embeds
        print('txt_embed', txt_embed, ' shape: ', txt_embed.shape)
        img_embed = model_outputs.image_embeds
        img_embed = img_embed.reshape(4, 4, -1)
        print('img_embed', img_embed)
        video_features_pooled = self.pooling(txt_embed, img_embed)
        print('video_features_pooled', video_features_pooled)
        logits = self.classifier(video_features_pooled)
        print('logits', logits)
        score = nn.functional.softmax(logits, dim=-1)
        print('score', score, ' shape: ', score.shape)
        # return logits, video_features_pooled, score, labels
        return logits, score, labels
    

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.embed_dim = args.embed_dim
        dropout = args.dropout_rate

        self.cross_attn = MHA(args)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_layer = nn.Linear(640, 512)
        self.proj_layer_2 = nn.Linear(384, 512)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(self.embed_dim, args.label_number)

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

        # print('video embeds shape: ', video_embeds.shape)
        # batch_size, num_frames, _ = video_embeds.shape
        # video_embeds = self.proj_layer(video_embeds.view(-1, 640)).view(batch_size, num_frames, 512)
        batch_size, num_frames, feature_size = video_embeds.shape
        #print('video embeds shape: ', batch_size)
        #print('video embeds shape: ', num_frames)
        #print('video embeds shape: ', feature_size)
        #assert feature_size * num_frames == 640, "Ensure total feature size is 640"
        #video_embeds = self.proj_layer(video_embeds.view(batch_size * num_frames, feature_size)).view(batch_size, num_frames, 512)

        if feature_size == 640:
            video_embeds = self.proj_layer(video_embeds.view(-1, 640)).view(batch_size, num_frames, 512)
        else:
            video_embeds = self.proj_layer_2(video_embeds.view(-1, 384)).view(batch_size, num_frames, 512)
        video_embeds = self.layer_norm1(video_embeds)
        # print('video embeds shape final: ', video_embeds.shape)

        # num_vids x num_texts x embed_dim
        #print('text embed transfromer: ', text_embeds.shape)
        #print('video embed transfromer: ', video_embeds.shape)
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out