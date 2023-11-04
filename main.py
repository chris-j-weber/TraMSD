import os
import argparse
import torch
import random
import numpy as np
import wandb
from models.model import TextModel, FusionModel, CrossAttentionModel
from models.engine import train, test

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    
    ## general
    parser.add_argument('--model', default='cross_attention', type=str, help='choose between text, fusion and cross_attention')
    parser.add_argument('--device', default='0', type=str, help='device')

    ## training
    parser.add_argument('--text_lr', default=None, type=float, help='learning rate for text parameters')
    parser.add_argument('--vision_lr', default=None, type=float, help='learning rate for vision parameters')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate for non clip parameters')
    parser.add_argument('--weight_decay', default=0.2, type=float, help='weight decay for regularization')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warmup proportion for learning rate scheduler')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout probability')

    ## model
    parser.add_argument('--pretrained_model', default="openai/clip-vit-base-patch32", type=str, help="load pretrained model")
    parser.add_argument('--freeze_pretrained_model', action="store_true", default=False)
    parser.add_argument('--num_train_epoches', default=2, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train and valid')
    parser.add_argument('--num_heads_ca', default=8, type=int, help='number of heads for cross attention')
    parser.add_argument('--label_number', default=2, type=int, help='number of labels')

    ## experiment
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    ## data
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--model_output_directory', default='models/checkpoints', type=str, help='folder where model is saved to')
    parser.add_argument('--path_to_pt', default='data/mustard/preprocessed/', type=str, help='path to .pt file')

    return parser.parse_args()
    
def main():
    args = set_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

    wandb.init(project='mmtsarcasm', notes='mmt', tags=['mmt'], config=vars(args))
    wandb.watch_called = False
    
    ## define learning rate
    if not args.text_lr:
        args.text_lr = args.lr

    if not args.vision_lr:
        args.vision_lr = args.lr
    
    ## define model
    if args.model == 'text':
        model = TextModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model_text.text_model.named_parameters():
                p.requires_grad = False
            for _, p in model.classification_head.named_parameters():
                p.requires_grad = True
    elif args.model == 'fusion':
        model = FusionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model_text.text_model.named_parameters():
                if 'final_layer_norm' in _ or 'layer_norm2' in _:
                    p.requires_grad = True
                else:    
                    p.requires_grad = False
            for _, p in model.model_vision.vision_model.named_parameters():
                if 'post_layernorm' in _ or 'layer_norm2' in _:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
    elif args.model == 'cross_attention':
        model = CrossAttentionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model_text.text_model.named_parameters():
                if 'final_layer_norm' in _ or 'layer_norm2' in _:
                    p.requires_grad = True
                else:    
                    p.requires_grad = False
            for _, p in model.model_vision.vision_model.named_parameters():
                if 'post_layernorm' in _ or 'layer_norm2' in _:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

    # print(model)
    # nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\nNumber of model parameters: {nb_params/1e6:.3f}M\n")

    model.to(device)
    wandb.watch(model, log='all')

    train(args, model, device)

    test(args, model, device)

if __name__ == '__main__':
    main()