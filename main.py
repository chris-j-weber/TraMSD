import os
import argparse
import torch
import random
import numpy as np
#import wandb
from data.dataset import Mustard
from models.CLIP.model import CLIP
from models.CLIP.train import train
from transformers import CLIPProcessor

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    
    #training
    parser.add_argument('--clip_lr', default=1e-6, type=float, help='learning rate for clip parameters')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate for non clip parameters')
    parser.add_argument('--num_train_epoches', default=6, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train and valid')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay for regularization')

    #model
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='max norm of clip parameter gradients')
    parser.add_argument('--dropout_rate', default=0.3, type=float, help='dropout probability')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warmup proportion for learning rate scheduler')
    parser.add_argument('--epsilon_adam', default=1e-8, type=float, help='epsilon for adam optimizer')
    parser.add_argument('--layers', default=3, type=int, help='number of transformer layers')
    parser.add_argument('--embed_dim', default=512, type=int, help='dimensionality of the model embedding')
    parser.add_argument('--num_mha', default=8, type=int, help='number of multi head attentions')

    #experiment
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=224, type=int, help='image hidden size')
    parser.add_argument('--text_max_len', default=77, type=int, help='max length of text for clip')
    parser.add_argument('--seed', default=24, type=int, help='random seed')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--device', default='0', type=str, help='device')
    parser.add_argument('--label_number', default=2, type=int, help='number of labels')

    #data
    parser.add_argument('--labels', default=2, type=int, help='number of labels')
    parser.add_argument('--videos_path', default='videos', type=str, help='path to videos')
    parser.add_argument('--num_frames', default=12, type=int, help='number of frames')
    parser.add_argument('--video_sample_type', default='uniform', type=str, help='video sample type')
    parser.add_argument('--input_res', default=224, type=int, help='input resolution for videos')
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='output path')

    #parser.add_argument('--', default=, type=, help='')

    return parser.parse_args()
    
def main():
    args = set_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
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

    #wandb.init(project='mmtsarcasm', notes='mmt', tags=['mmt'], config=vars(args))
    #wandb.watch_called = False

    train_data = Mustard(mode='train')
    val_data = Mustard(mode='val')
    test_data = Mustard(mode='test')

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIP(args)

    model.to(device)
    #wandb.watch(model, log='all')

    train(args, train_data, val_data, test_data, model, processor, device)

if __name__ == '__main__':
    main()