import torch
import torch.nn as nn
from transformers import CLIPModel


class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.args = args

        self.model = CLIPModel.from_pretrained(args.pretrained_model)

        self.text_embed_dim = self.model.text_embed_dim

        self.classification_head = nn.Linear(self.text_embed_dim, self.args.label_number)

    def forward(self, inputs):
        model_outputs = self.model(**inputs, output_attentions=True)

        # get text cls token
        text_model_output = model_outputs.text_model_output
        text_cls_token = text_model_output['pooler_output']
        
        # classification head
        output = self.classification_head(text_cls_token)        

        # return logits
        return output


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        self.args = args

        self.model = CLIPModel.from_pretrained(args.pretrained_model)

        self.text_embed_dim = self.model.text_embed_dim
        self.vision_embed_dim = self.model.vision_embed_dim
        self.fusion_embed_dim = self.text_embed_dim + self.vision_embed_dim

        self.classification_head = nn.Linear(self.fusion_embed_dim, self.args.label_number)

    def forward(self, inputs):
        model_outputs = self.model(**inputs, output_attentions=True)

        # get text cls token
        text_model_output = model_outputs.text_model_output
        text_cls_token = text_model_output['pooler_output']

        # get vision cls token
        vision_model_output = model_outputs.vision_model_output
        batch_size = text_model_output[0].shape[0]
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1]
        vision_model_output_video = vision_model_output[0].reshape(batch_size, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] # cls token of all frames

        num_frames = vision_model_output_video.shape[1]
        # average cls token of all frames per video
        video_cls_token = frames_cls_token.sum(dim=1) / num_frames 

        # concatenate text cls token and video cls token
        fusion_token = torch.cat((text_cls_token, video_cls_token), dim=1)
        
        # classification head
        output = self.classification_head(fusion_token)        

        # return logits
        return output
    

class CrossAttentionModel(nn.Module):
    def __init__(self, args):
        super(CrossAttentionModel, self).__init__()
        self.args = args

        self.model = CLIPModel.from_pretrained(args.pretrained_model)

        self.text_embed_dim = self.model.text_embed_dim
        self.vision_embed_dim = self.model.vision_embed_dim

        self.vision_projector = nn.Linear(self.vision_embed_dim, self.text_embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.text_embed_dim, num_heads=self.args.num_heads_ca, batch_first=True)

        self.classification_head = nn.Linear(self.text_embed_dim, self.args.label_number)

    def forward(self, inputs):
        model_outputs = self.model(**inputs, output_attentions=True)

        # get text cls token
        text_model_output = model_outputs.text_model_output
        text_cls_token = text_model_output['pooler_output'].unsqueeze(dim=1)

        # get vision cls token
        vision_model_output = model_outputs.vision_model_output
        batch_size = text_model_output[0].shape[0]
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1]
        vision_model_output_video = vision_model_output[0].reshape(batch_size, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] # cls token of all frames

        # compute cross attention between text cls token (query) and frame cls tokens (key, value)
        frames_cls_token = torch.flatten(frames_cls_token, start_dim=0, end_dim=1)
        projected_frames_cls_token = self.vision_projector(frames_cls_token)
        projected_frames_cls_token = projected_frames_cls_token.reshape(batch_size, -1, self.text_embed_dim)

        ca_token, _ = self.cross_attention(text_cls_token, projected_frames_cls_token, projected_frames_cls_token)
        ca_token = ca_token.squeeze()

        # classification head
        output = self.classification_head(ca_token)        

        # return logits
        return output