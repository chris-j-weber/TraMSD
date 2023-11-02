import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel
from utils.utils import unflatten, text_mean_pooling

class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.args = args

        self.model_text = CLIPTextModel.from_pretrained(args.pretrained_model, output_attentions=False)

        self.text_embed_dim = self.model_text.config.hidden_size

        self.classification_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )
    
    def forward(self, text_input, lengths):
        model_outputs = self.model_text(**text_input)
        
        ## get average token per text (of a video)
        avg_token_per_sentence = text_mean_pooling(model_outputs, text_input['attention_mask'])
        avg_token_per_video = unflatten(avg_token_per_sentence, lengths)

        ## classification head
        output = self.classification_head(avg_token_per_video)   

        return output


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        self.args = args

        self.model_text = CLIPTextModel.from_pretrained(args.pretrained_model, output_attentions=False)
        self.model_vision = CLIPVisionModel.from_pretrained(args.pretrained_model, output_attentions=False)

        self.text_embed_dim = self.model_text.config.hidden_size
        self.vision_embed_dim = self.model_vision.config.hidden_size
        self.fusion_embed_dim = self.text_embed_dim + self.vision_embed_dim

        self.classification_head = nn.Sequential(
            nn.Linear(self.fusion_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, text_input, video_input, lengths):  
        text_model_output = self.model_text(**text_input, output_attentions=False)
        vision_model_output = self.model_vision(**video_input, output_attentions=False)

        ## get average token per text (of a video)
        avg_token_per_sentence = text_mean_pooling(text_model_output, text_input['attention_mask'])
        avg_token_per_text = unflatten(avg_token_per_sentence, lengths)

        ## get frames cls tokens
        batch_size = len(lengths)
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1] 
        vision_model_output_video = vision_model_output[0].reshape(batch_size, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] ## cls token of all frames

        ## average cls token of all frames (per video)
        num_frames = vision_model_output_video.shape[1]
        video_cls_token = frames_cls_token.sum(dim=1) / num_frames 

        ## concatenate text tokens and frame tokens
        fusion_token = torch.cat((avg_token_per_text, video_cls_token), dim=1)
        
        ## classification head
        output = self.classification_head(fusion_token) 

        return output
    

class CrossAttentionModel(nn.Module):
    def __init__(self, args):
        super(CrossAttentionModel, self).__init__()
        self.args = args

        self.model_text = CLIPTextModel.from_pretrained(args.pretrained_model, output_attentions=False)
        self.model_vision = CLIPVisionModel.from_pretrained(args.pretrained_model, output_attentions=False)

        self.text_embed_dim = self.model_text.config.hidden_size
        self.vision_embed_dim = self.model_vision.config.hidden_size

        self.vision_projector = nn.Linear(self.vision_embed_dim, self.text_embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.text_embed_dim, 
                                                     num_heads=self.args.num_heads_ca, 
                                                     batch_first=True)

        self.classification_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, text_input, video_input, lengths):
        text_model_output = self.model_text(**text_input, output_attentions=False)
        vision_model_output = self.model_vision(**video_input, output_attentions=False)

        ## get average token per text (of a video)
        avg_token_per_sentence = text_mean_pooling(text_model_output, text_input['attention_mask'])
        avg_token_per_video = unflatten(avg_token_per_sentence, lengths)
        avg_token_per_video = avg_token_per_video.unsqueeze(dim=1)

        ## get frames cls tokens
        batch_size = len(lengths)
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1]
        vision_model_output_video = vision_model_output[0].reshape(batch_size, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] ## cls token of all frames

        ## compute cross attention between text tokens (query) and frame tokens (key, value)
        frames_cls_token = torch.flatten(frames_cls_token, start_dim=0, end_dim=1)
        projected_frames_cls_token = self.vision_projector(frames_cls_token)
        projected_frames_cls_token = projected_frames_cls_token.reshape(batch_size, -1, self.text_embed_dim)

        ca_token, _ = self.cross_attention(avg_token_per_video, projected_frames_cls_token, projected_frames_cls_token)
        ca_token = ca_token.squeeze()

        ## classification head
        output = self.classification_head(ca_token)        

        return output