import os
import argparse
import torch
import random
import wandb
import numpy as np
from data.dataset import Mustard
from models.model import TextModel, FusionModel, CrossAttentionModel
from models.engine import train, test
from transformers import CLIPProcessor

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    
    # general
    parser.add_argument('--model', default='text', type=str, help='choose between text, fusion and cross_attention')
    parser.add_argument('--device', default='0', type=str, help='device')

    # training
    parser.add_argument('--clip_lr', default=1e-6, type=float, help='learning rate for clip parameters')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate for non clip parameters')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay for regularization')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warmup proportion for learning rate scheduler')

    # model
    parser.add_argument('--pretrained_model', default='openai/clip-vit-base-patch32', type=str, help='load pretrained model')
    parser.add_argument('--freeze_pretrained_model', action='store_true', default=False, help='freeze pretrained model')
    parser.add_argument('--num_train_epoches', default=3, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train and valid')
    parser.add_argument('--num_heads_ca', default=8, type=int, help='number of heads for cross attention')
    parser.add_argument('--label_number', default=2, type=int, help='number of labels')

    # old model
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='max norm of clip parameter gradients')
    parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout probability')
    parser.add_argument('--epsilon_adam', default=1e-8, type=float, help='epsilon for adam optimizer')
    # parser.add_argument('--layers', default=3, type=int, help='number of transformer layers')
    parser.add_argument('--frame_batch_size', default=4, type=int, help='batch size of frames')

    # experiment
    # parser.add_argument('--text_max_len', default=77, type=int, help='max length of text for clip')
    parser.add_argument('--seed', default=24, type=int, help='random seed')

    # data
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--label_number', default=2, type=int, help='number of labels')
    # parser.add_argument('--videos_path', default='videos', type=str, help='path to videos')
    # parser.add_argument('--num_frames', default=12, type=int, help='number of frames')
    # parser.add_argument('--video_sample_type', default='uniform', type=str, help='video sample type')
    # parser.add_argument('--input_res', default=224, type=int, help='input resolution for videos')
    parser.add_argument('--model_output_directory', default='models/checkpoints', type=str, help='folder where model is saved to')
    parser.add_argument('--optimizer_output_directory', default='./optimizer_output', type=str, help='folder where optimizer is saved to')
    parser.add_argument('--logging_directory', default='./logging', type=str, help='logging path')

    #parser.add_argument('--', default=, type=, help='')

    return parser.parse_args()
    
def main():
    args = set_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    if not os.path.exists(args.logging_directory):
        os.mkdir(args.logging_directory)
    else:
        os.rmdir(args.logging_directory)
        os.mkdir(args.logging_directory)

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

    train_data = Mustard(mode='train')
    val_data = Mustard(mode='val')
    test_data = Mustard(mode='test')

    processor = CLIPProcessor.from_pretrained(args.pretrained_model)

    if args.model == 'text':
        model = TextModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.classification_head.named_parameters():
                p.requires_grad = True
    elif args.model == 'fusion':
        model = FusionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model.text_model.named_parameters():
                p.requires_grad = False
            for _, p in model.model.vision_model.named_parameters():
                p.requires_grad = False
    elif args.model == 'cross_attention':
        model = CrossAttentionModel(args)
        if args.freeze_pretrained_model == True:
            for _, p in model.model.text_model.named_parameters():
                p.requires_grad = False
            for _, p in model.model.vision_model.named_parameters():
                p.requires_grad = False

    model.to(device)
    wandb.watch(model, log='all')

    train(args, model, device, train_data, val_data, processor)
    
    test(args, model, device, test_data, processor)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()