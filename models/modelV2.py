import torch.nn as nn
from transformers import CLIPModel
from models.mha import MultiHeadAttention

class CLIPTransformer(nn.Module):
    def __init__(self, args):
        super(CLIPTransformer, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.embed_dim = args.embed_dim
        dropout = args.dropout_rate

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
        
        img_embed = model_outputs.image_embeds
        img_embed = img_embed.reshape(4, 4, -1)
        
        video_features_pooled = self.pooling(txt_embed, img_embed)
        logits = self.classifier(video_features_pooled)
        score = nn.functional.softmax(logits, dim=-1)
        
        return logits, score, labels
    

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.embed_dim = args.embed_dim
        dropout = args.dropout_rate

        self.cross_attn = MHA(args)
        self.cross_attention = MultiHeadAttention(args)

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

        batch_size, num_frames, feature_size = video_embeds.shape

        if feature_size == 640:
            video_embeds = self.proj_layer(video_embeds.view(-1, 640)).view(batch_size, num_frames, 512)
        else:
            video_embeds = self.proj_layer_2(video_embeds.view(-1, 384)).view(batch_size, num_frames, 512)
        video_embeds = self.layer_norm1(video_embeds)

        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out