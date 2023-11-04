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
    parser.add_argument('--num_train_epoches', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for train and valid')
    parser.add_argument('--num_heads_ca', default=8, type=int, help='number of heads for cross attention')
    parser.add_argument('--label_number', default=2, type=int, help='number of labels')

    ## experiment
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    ## data
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
    parser.add_argument('--model_output_directory', default='model/checkpoints', type=str, help='folder where model is saved to')
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

    ## define learning rates
    if not args.text_lr:
        args.text_lr = args.lr
    if not args.vision_lr:
        args.vision_lr = args.lr

    if args.model == 'text':
        model = TextModel(args)
    elif args.model == 'fusion':
        model = FusionModel(args)
    elif args.model == 'cross_attention':
        model = CrossAttentionModel(args)

    model.to(device)
    wandb.watch(model, log='all')

    ## load modul processors
    text_processor = CLIPTokenizerFast.from_pretrained(args.pretrained_model)
    vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model)

    ## load model checkpoint
    load_file = os.path.join(args.model_output_directory, 'checkpoint_{args.model}.pth')
    checkpoint = torch.load(load_file, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    ## define optimizer
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
    
    ## check and load optimizer if possible
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    criterion = nn.CrossEntropyLoss()

    ## load test data
    if args.model in ['fusion', 'cross_attention']:
        text_data = MustardVideoText(args, device, args.path_to_pt+'video_test.pt', args.path_to_pt+'text_test.pt', args.path_to_pt+'labels_test.pt', args.path_to_pt+'ids_test.pt')
        test_loader = DataLoader(text_data, batch_size=args.batch_size, collate_fn=MustardVideoText.collate_func, shuffle=False, num_workers=args.num_workers)
    else:
        text_data = MustardText(args, device, args.path_to_pt+'text_test.pt', args.path_to_pt+'labels_test.pt')
        test_loader = DataLoader(text_data, batch_size=args.batch_size, collate_fn=MustardText.collate_func, shuffle=False, num_workers=args.num_workers)

    ## run metric evaluation
    loss, acc, f1, auc, pre, rec = evaluate(args, model, device, criterion, test_loader, text_processor, vision_processor)

    ## test results
    wandb.log({f'test_loss': loss, 
               f'test_acc': acc, 
               f'test_f1': f1, 
               f'test_auc': auc, 
               f'test_pre': pre, 
               f'test_rec': rec})
    logging.info('test_loss is {}, test_acc is {}, test_f1 is {}, test_auc is {}, test_pre is {}, test_rec is {}'.format(loss, acc, f1, auc, pre, rec))

    logger.info('Test done')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()