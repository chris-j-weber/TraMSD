import os
import wandb
import logging
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers import CLIPTokenizerFast, CLIPImageProcessor

from utils.metrics import evaluate
from data.dataset import MustardVideoText, MustardText
from models.model import TextModel, FusionModel, CrossAttentionModel

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser()
    
    ## general
    parser.add_argument('--model', default='fusion', type=str, help='choose between text, fusion and cross_attention')
    parser.add_argument('--device', default='0', type=str, help='device')

    ## data
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

    wandb.init(project='mmtsarcasm', notes='mmt', tags=['mmt'], config=vars(args))
    wandb.watch_called = False

    if args.model == 'text':
        model = TextModel(args)
    elif args.model == 'fusion':
        model = FusionModel(args)
    elif args.model == 'cross_attention':
        model = CrossAttentionModel(args)

    model.to(device)
    wandb.watch(model, log='all')

    text_processor = CLIPTokenizerFast.from_pretrained(args.pretrained_model)
    vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model)

    load_file = os.path.join(args.model_output_directory, 'checkpoint_{args.model}.pth')
    checkpoint = torch.load(load_file, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if args.model in ['fusion', 'cross_attention']:
        base_params = [param for name, param in model.named_parameters() if 'model_vision' not in name and 'model_text' not in name]
        optimizer = AdamW([{'params': base_params},
                           {'params': model.model_vision.parameters(), 'lr': args.vision_lr},
                           {'params': model.model_text.parameters(), 'lr': args.text_lr}], 
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    else:
        base_params = [param for name, param in model.named_parameters() if 'model_text' not in name]
        optimizer = AdamW([{'params': base_params},
                           {'params': model.model_text.parameters(), 'lr': args.text_lr}], 
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    criterion = nn.CrossEntropyLoss()

    if args.model in ['fusion', 'cross_attention']:
        text_data = MustardVideoText(args, device, args.path_to_pt+'video_test.pt', args.path_to_pt+'text_test.pt', args.path_to_pt+'labels_test.pt', args.path_to_pt+'ids_test.pt')
        test_loader = DataLoader(text_data, batch_size=args.batch_size, collate_fn=MustardVideoText.collate_func, shuffle=False, num_workers=args.num_workers)
    else:
        text_data = MustardText(args, device, args.path_to_pt+'text_test.pt', args.path_to_pt+'labels_test.pt')
        test_loader = DataLoader(text_data, batch_size=args.batch_size, collate_fn=MustardText.collate_func, shuffle=False, num_workers=args.num_workers)

    epoch_loss, acc, f1, auc = evaluate(args, model, device, criterion, test_loader, text_processor, vision_processor)

    ## test results
    wandb.log({f'test_loss_{args.model}': epoch_loss, f'test_acc_{args.model}': acc, f'test_f1_{args.model}': f1, f'test_auc_{args.model}': auc})
    logging.info('test_loss is {}, test_acc is {}, test_f1 is {}, test_auc is {}'.format(epoch_loss, acc, f1, auc))

    logger.info('Test done')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()