import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer

class Encoder(nn.Module):
    def __init__(self, config, layer_number):
        super(Encoder, self).__init__()
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
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512
        self.config.num_attention_heads = args.num_mha
        self.encoding = Encoder(self.config, layer_number=args.layers)

        self.text_linear =  nn.Sequential(
            nn.Linear(args.text_size, args.text_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )
        self.image_linear =  nn.Sequential(
            nn.Linear(768, args.image_size),
            nn.Dropout(args.dropout_rate),
            nn.GELU()
        )

        self.classifier_fuse = nn.Linear(args.text_size , args.label_number)
        self.classifier_text = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, labels):
        modeling_outputs = self.model(**inputs,output_attentions=True)
        
        txt_features, img_features, txt_cls, img_cls = self.extract_features(modeling_outputs)
        input_embeddings, extended_attention_mask = self.build_attention(inputs, txt_features, img_features)
        fused_feature = self.extract_fused_features(inputs, input_embeddings, extended_attention_mask)

        logits_fused = self.classifier_fuse(fused_feature)
        logits_text = self.classifier_text(txt_cls)
        #logits_image = self.classifier_image(img_cls)
   
        fuse_score = nn.functional.softmax(logits_fused, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        #image_score = nn.functional.softmax(logits_image, dim=-1)

        #score = fuse_score + text_score + image_score
        score = fuse_score + text_score

        #outputs = (score, logits_fused, logits_text, logits_image, labels)
        outputs = (score, logits_fused, logits_text, labels)
        
        return outputs
    
    def extract_features(self, modeling_outputs):
        # extract image and text sequences at the output of the last layer
        text_features = modeling_outputs['text_model_output']['last_hidden_state']
        image_features = modeling_outputs['vision_model_output']['last_hidden_state']
        
        # extract classification tokens (after processing through a linear layer and a tanh activation)
        text_cls = modeling_outputs['text_model_output']['pooler_output']
        image_cls = modeling_outputs['vision_model_output']['pooler_output']
        
        # apply linear function 
        text_cls = self.text_linear(text_cls)
        image_cls = self.image_linear(image_cls)
        
        return text_features, image_features, text_cls, image_cls
    
    def build_attention(self, inputs, txt, img):
        # extract hidden state embeddings
        txt_embeddings = self.model.text_projection(txt)
        img_embeddings = self.model.visual_projection(img)
        
        # concatenate both embeddings
        input_embeddings = torch.cat((img_embeddings, txt_embeddings), dim=1)
        
        # build attention mask
        attention_mask = torch.cat((torch.ones(txt.shape[0], 50).to(txt.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return input_embeddings, extended_attention_mask
    
    def extract_fused_features(self, inputs, embeddings, attentions):
        # encode concatenated embeddings
        fused_hidden_states, all_attentions = self.encoding(embeddings, attentions, output_all_encoded_layers=False)
        fused_hidden_states = fused_hidden_states[-1]
       
        # extract new features
        new_text_features = fused_hidden_states[:, 50:, :]
        new_text_feature = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]
        new_image_feature = fused_hidden_states[:, 0, :].squeeze(1)

        # apply key-less attention
        text_weight = self.att(new_text_feature)
        image_weight = self.att(new_image_feature)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)

        # fused cls feature
        fuse_feature = tw.squeeze(1) * new_text_feature + iw.squeeze(1) * new_image_feature
       
        return fuse_feature