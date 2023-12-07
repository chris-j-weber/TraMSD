import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPVisionModel
from utils.utils import unflatten, text_mean_pooling, text_max_pooling, attention_pooling

class TextModel(nn.Module):
    def __init__(self, args):
        super(TextModel, self).__init__()
        self.args = args

        self.model_text = CLIPTextModel.from_pretrained(args.pretrained_model, output_attentions=False)

        self.text_embed_dim = self.model_text.config.hidden_size

        self.attention_pooling = nn.MultiheadAttention(embed_dim=self.text_embed_dim, 
                                                       num_heads=self.args.num_heads_ca, 
                                                       batch_first=True)

        self.classification_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, text_input, lengths):
        model_outputs = self.model_text(**text_input)
        
        if self.args.pooling == "max_attention":
            ## max pooling -> attention pooling
            token_per_sentence = text_max_pooling(model_outputs, text_input['attention_mask'])
            token_per_dialogue = self.attention_pool(token_per_sentence, lengths)
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        elif self.args.pooling == "mean_attention":
            ## mean pooling -> attention pooling
            token_per_sentence = text_mean_pooling(model_outputs, text_input['attention_mask'])
            token_per_dialogue = self.attention_pool(token_per_sentence, lengths)
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        else:
            ## mean pooling -> mean pooling
            token_per_sentence = text_mean_pooling(model_outputs, text_input['attention_mask'])
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        
        ## classification head
        output = self.classification_head(token_per_dialogue)

        return output
    
    def attention_pool(self, max_pooled_embeddings, original_lengths):
        
        pooled_results = []
        start_idx = 0

        for length in original_lengths:
            segment = max_pooled_embeddings[start_idx:start_idx + length]
            query = torch.mean(segment, dim=0, keepdim=True)
        
            att_token, _ = self.attention_pooling(query, segment, segment)
            pooled_results.append(att_token)
            start_idx += length
    
        pooled_tensor = torch.stack(pooled_results, dim=0)

        return pooled_tensor.squeeze()


class FusionModel(nn.Module):
    def __init__(self, args):
        super(FusionModel, self).__init__()
        self.args = args

        self.model_text = CLIPTextModel.from_pretrained(args.pretrained_model, output_attentions=False)
        self.model_vision = CLIPVisionModel.from_pretrained(args.pretrained_model, output_attentions=False)

        self.text_embed_dim = self.model_text.config.hidden_size
        self.vision_embed_dim = self.model_vision.config.hidden_size
        self.fusion_embed_dim = self.text_embed_dim + self.vision_embed_dim

        self.attention_pooling = nn.MultiheadAttention(embed_dim=self.text_embed_dim, 
                                                       num_heads=self.args.num_heads_ca, 
                                                       batch_first=True)
        
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
        if self.args.pooling == "max_attention":
            ## max pooling -> attention pooling
            token_per_sentence = text_max_pooling(text_model_output, text_input['attention_mask'])
            token_per_dialogue = self.attention_pool(token_per_sentence, lengths)
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        elif self.args.pooling == "mean_attention":
            ## mean pooling -> attention pooling
            token_per_sentence = text_mean_pooling(text_model_output, text_input['attention_mask'])
            token_per_dialogue = self.attention_pool(token_per_sentence, lengths)
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        else:
            ## mean pooling -> mean pooling
            token_per_sentence = text_mean_pooling(text_model_output, text_input['attention_mask'])
            token_per_dialogue = unflatten(token_per_sentence, lengths)

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
        fusion_token = torch.cat((token_per_dialogue, video_cls_token), dim=1)
        
        ## classification head
        output = self.classification_head(fusion_token) 

        return output
    
    def attention_pool(self, max_pooled_embeddings, original_lengths):
        
        pooled_results = []
        start_idx = 0

        for length in original_lengths:
            segment = max_pooled_embeddings[start_idx:start_idx + length]
            query = torch.mean(segment, dim=0, keepdim=True)
        
            att_token, _ = self.attention_pooling(query, segment, segment)
            pooled_results.append(att_token)
            start_idx += length
    
        pooled_tensor = torch.stack(pooled_results, dim=0)

        return pooled_tensor.squeeze()
    

class CrossAttentionModel(nn.Module):
    def __init__(self, args):
        super(CrossAttentionModel, self).__init__()
        self.args = args

        self.model_text = CLIPTextModel.from_pretrained(args.pretrained_model, output_attentions=False)
        self.model_vision = CLIPVisionModel.from_pretrained(args.pretrained_model, output_attentions=False)

        self.text_embed_dim = self.model_text.config.hidden_size
        self.vision_embed_dim = self.model_vision.config.hidden_size

        self.attention_pooling = nn.MultiheadAttention(embed_dim=self.text_embed_dim, 
                                                       num_heads=self.args.num_heads_ca, 
                                                       batch_first=True)
        
        self.text_projector = nn.Linear(self.text_embed_dim, self.vision_embed_dim)
        self.vision_projector = nn.Linear(self.vision_embed_dim, self.text_embed_dim)
        # self.cross_attention = nn.MultiheadAttention(embed_dim=self.text_embed_dim, 
        #                                              num_heads=self.args.num_heads_ca, 
        #                                              batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.vision_embed_dim, 
                                                     num_heads=self.args.num_heads_ca, 
                                                     batch_first=True)

        # self.classification_head = nn.Linear(self.text_embed_dim, self.args.label_number)
        self.classification_head = nn.Sequential(
            nn.Linear(self.vision_embed_dim, 256),
            # nn.Linear(self.text_embed_dim, 256),
            nn.Dropout(p=args.dropout_rate),
            nn.ReLU(),
            nn.Linear(256, self.args.label_number)
        )

    def forward(self, text_input, video_input, lengths):
        text_model_output = self.model_text(**text_input, output_attentions=False)
        vision_model_output = self.model_vision(**video_input, output_attentions=False)

        ## get average token per text (of a video)
        if self.args.pooling == "max_attention":
            ## max pooling -> attention pooling
            token_per_sentence = text_max_pooling(text_model_output, text_input['attention_mask'])
            token_per_dialogue = self.attention_pool(token_per_sentence, lengths)
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        elif self.args.pooling == "mean_attention":
            ## mean pooling -> attention pooling
            token_per_sentence = text_mean_pooling(text_model_output, text_input['attention_mask'])
            token_per_dialogue = self.attention_pool(token_per_sentence, lengths)
            token_per_dialogue = unflatten(token_per_sentence, lengths)
        else:
            ## mean pooling -> mean pooling
            token_per_sentence = text_mean_pooling(text_model_output, text_input['attention_mask'])
            token_per_dialogue = unflatten(token_per_sentence, lengths)
            
        token_per_dialogue = token_per_dialogue.unsqueeze(dim=1)
        projected_token_per_dialogue = self.text_projector(token_per_dialogue)

        ## get frames cls tokens
        batch_size = len(lengths)
        nb_tokens = vision_model_output[0].shape[-2]
        embed_dim = vision_model_output[0].shape[-1]
        vision_model_output_video = vision_model_output[0].reshape(batch_size, -1, nb_tokens, embed_dim)
        frames_cls_token = vision_model_output_video[..., 0, :] ## cls token of all frames

        ## compute cross attention between text tokens (query) and frame tokens (key, value)
        projected_frames_cls_token = frames_cls_token.reshape(batch_size, -1, self.vision_embed_dim)

        ca_token, _ = self.cross_attention(projected_token_per_dialogue, projected_frames_cls_token, projected_frames_cls_token)
        ca_token = ca_token.squeeze()

        ## classification head
        output = self.classification_head(ca_token)        

        return output
    
    def attention_pool(self, max_pooled_embeddings, original_lengths):
        
        pooled_results = []
        start_idx = 0

        for length in original_lengths:
            segment = max_pooled_embeddings[start_idx:start_idx + length]
            query = torch.mean(segment, dim=0, keepdim=True)
        
            att_token, _ = self.attention_pooling(query, segment, segment)
            pooled_results.append(att_token)
            start_idx += length
    
        pooled_tensor = torch.stack(pooled_results, dim=0)

        return pooled_tensor.squeeze()