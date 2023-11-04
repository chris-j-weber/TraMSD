import os
import wandb
import logging
import torch
import numpy as np
import torch.nn as nn
from data.dataset import MustardVideoText, MustardText
from torch.utils.data import DataLoader
from tqdm import trange
from sklearn import metrics
from transformers import CLIPTokenizerFast, CLIPImageProcessor
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.metrics import evaluate
from utils.utils import flatten

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args, model, device):
    if not os.path.exists(args.model_output_directory):
        os.mkdir(args.model_output_directory)
    
    text_processor = CLIPTokenizerFast.from_pretrained(args.pretrained_model)
    vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model)

    if args.model in ['fusion', 'cross_attention']:
        train_data = MustardVideoText(args, device, args.path_to_pt+'video_train.pt', args.path_to_pt+'text_train.pt', args.path_to_pt+'labels_train.pt', args.path_to_pt+'ids_train.pt')
        val_data = MustardVideoText(args, device, args.path_to_pt+'video_val.pt', args.path_to_pt+'text_val.pt', args.path_to_pt+'labels_val.pt', args.path_to_pt+'ids_val.pt')
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=MustardVideoText.collate_func, shuffle=True, num_workers=args.num_workers)
    else:
        train_data = MustardText(args, device, args.path_to_pt+'text_train.pt', args.path_to_pt+'labels_train.pt')
        val_data = MustardText(args, device, args.path_to_pt+'text_val.pt', args.path_to_pt+'labels_val.pt')
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=MustardText.collate_func, shuffle=True, num_workers=args.num_workers)

    total_steps = int(len(train_loader) * args.num_train_epoches)
    model.to(device)

    ## custom AdamW optimizer
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
    
    ## default AdamW optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), 
    #                               lr=args.lr, 
    #                               weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=6)

    max_accuracy = 0.

    for i_epoch in trange(0, int(args.num_train_epoches), desc='epoch', disable=False):
        model.train()

        prob = []
        y_pred = []
        targets = []
        running_loss = 0

        for step, batch in enumerate(train_loader):
            if args.model in ['fusion', 'cross_attention']:
                videos, text, label = batch

                ## flatten frames (set_size, num_frames, embedding_dim) -> (set_size*num_frames, embedding_dim)
                frames = torch.flatten(videos, start_dim=0, end_dim=1).to(device)

                ## flatten text from List[List[Strings]] -> List[Strings]
                flattened_text, text_lengths = flatten(text)
                
                video_inputs = vision_processor(images=frames, padding=True, truncation=True, return_tensors='pt').to(device)
            else:
                text, label = batch
                flattened_text, text_lengths = flatten(text)
            
            text_inputs = text_processor(text=flattened_text, padding=True, truncation=True, return_tensors='pt').to(device)

            target = torch.tensor(label).to(device)
            targets.extend(label)

            if args.model in ['fusion', 'cross_attention']:
                pred = model(text_inputs, video_inputs, text_lengths)
            else:
                pred = model(text_inputs, text_lengths)
            prob.extend(torch.nn.functional.softmax(pred, dim=-1).detach().cpu())

            loss = criterion(pred, target)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # stats
        epoch_loss = running_loss / len(train_loader)
        
        targets = torch.tensor(targets).to(device)
        y_pred = np.argmax(np.array(prob), axis=-1)
        acc = metrics.accuracy_score(targets.cpu(), y_pred)
        auc = metrics.roc_auc_score(targets.cpu(), np.array(prob)[:, 1])
        f1 = metrics.f1_score(targets.cpu(), y_pred, pos_label=1)
        precision = metrics.precision_score(targets.cpu(), y_pred)
        recall = metrics.recall_score(targets.cpu(), y_pred)

        ## train results
        wandb.log({f'train_loss': epoch_loss, 
                   f'train_acc': acc, 
                   f'train_f1': f1, 
                   f'train_auc': auc, 
                   f'train_pre': precision, 
                   f'train_rec': recall})
        logging.info('i_epoch is {}, train_loss is {}, train_acc is {}, train_f1 is {}, train_auc is {}, train_pre is {}, train_rec is {}'.format(i_epoch, epoch_loss, acc, f1, auc, precision, recall))

        ## validation results
        validation_acc = validate(args, model, device, val_data, text_processor, vision_processor)

        ## scheduler for on plateau + new val returns
        # validation_acc, val_loss = validate(args, model, device, val_data, text_processor, vision_processor)
        # scheduler.step(val_loss)

        ## save best model
        if validation_acc > max_accuracy:
            max_accuracy = validation_acc

            if not os.path.exists(args.model_output_directory):
                os.mkdir(args.model_output_directory)

            checkpoint_file = f'checkpoint_{args.model}.pth'

            dict_to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }

            output_dir = os.path.join(args.model_output_directory, checkpoint_file)
            torch.save(dict_to_save, output_dir)

    logger.info('Train done')

def validate(args, model, device, val_data, text_processor, vision_processor):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    if args.model in ['fusion', 'cross_attention']:
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=MustardVideoText.collate_func, shuffle=False, num_workers=args.num_workers)
    else:
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=MustardText.collate_func, shuffle=False, num_workers=args.num_workers)

    val_loss, val_acc, val_f1, val_auc, val_pre, val_rec = evaluate(args, model, device, criterion, val_loader, text_processor, vision_processor)

    wandb.log({f'val_loss': val_loss,
               f'val_acc': val_acc, 
               f'val_f1': val_f1, 
               f'val_auc': val_auc, 
               f'val_pre': val_pre, 
               f'val_rec': val_rec})
    logging.info('val_loss is {}, val_acc is {}, val_f1 is {}, val_auc is {}, val_pre is {}, val_rec is {}'.format(val_loss, val_acc, val_f1, val_auc, val_pre, val_rec))

    return val_acc
    # return val_acc, val_loss

def test(args, model, device):

    text_processor = CLIPTokenizerFast.from_pretrained(args.pretrained_model)
    vision_processor = CLIPImageProcessor.from_pretrained(args.pretrained_model)

    load_file = os.path.join(args.model_output_directory, f'checkpoint_{args.model}.pth')
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

    loss, acc, f1, auc, pre, rec = evaluate(args, model, device, criterion, test_loader, text_processor, vision_processor)

    ## test results
    wandb.log({f'test_loss': loss,
               f'test_acc': acc, 
               f'test_f1': f1, 
               f'test_auc': auc, 
               f'test_pre': pre, 
               f'test_rec': rec})
    logging.info('test_loss is {}, test_acc is {}, test_f1 is {}, test_auc is {}, test_pre is {}, test_rec is {}'.format(loss, acc, f1, auc, pre, rec))

    # torch.cuda.empty_cache()
    logger.info('Test done')