import os
import wandb
import logging
import torch
import numpy as np
import torch.nn as nn
from data.dataset import Mustard
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn import metrics
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils.metrics import evaluate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args, model, device, train_data, val_data, processor):
    if not os.path.exists(args.model_output_directory):
        os.mkdir(args.model_output_directory)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=True, num_workers=args.num_workers)
    total_steps = int(len(train_loader) * args.num_train_epoches)
    model.to(device)

    # clip_params = list(map(id, model.model.parameters()))
    # base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    # optimizer = AdamW([
    #         {'params': base_params},
    #         {'params': model.model.parameters(),'lr': args.clip_lr}
    #         ], lr=args.lr, weight_decay=args.weight_decay)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)

    max_accuracy = 0.

    for i_epoch in trange(0, int(args.num_train_epoches), desc='epoch', disable=False):
        
        iteration_bar = tqdm(train_loader, desc='iteration ', disable=False)
        model.train()

        prob = []
        y_pred = []
        running_loss = 0

        for step, batch in enumerate(iteration_bar):
            text, videos, label, id_list = batch

            frames = torch.flatten(videos, start_dim=0, end_dim=1)
            inputs = processor(text=text, images=frames, padding=True, truncation=True, return_tensors='pt').to(device)
            target = torch.tensor(label).to(device)

            pred = model(inputs)
            prob.extend(torch.nn.functional.softmax(pred, dim=-1).detach().cpu())

            loss = criterion(pred, target)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            scheduler.step() 

            running_loss += loss.item()

        # stats
        epoch_loss = running_loss / len(train_loader)

        y_pred = np.argmax(np.array(prob), axis=-1)
        acc = metrics.accuracy_score(target.cpu(), y_pred)
        auc = metrics.roc_auc_score(target.cpu(), np.array(prob)[:, 1])
        f1 = metrics.f1_score(target.cpu(), y_pred, pos_label=1)

        # train results
        wandb.log({'train_loss': epoch_loss, 'train_acc': acc, 'train_f1': f1, 'train_auc': auc})
        logging.info('i_epoch is {}, train_loss is {}, train_acc is {}, train_f1 is {}, train_auc is {}'.format(i_epoch, epoch_loss, acc, f1, auc))

        # validation results
        val_epoch, validation_acc, validation_f1, validation_auc = evaluate(args, model, device, val_data, processor)
        wandb.log({'validation_acc': validation_acc, 'validation_f1': validation_f1, 'validation_auc': validation_auc})
        logging.info('i_epoch is {}, validation_acc is {}, validation_f1 is {}, validation_auc is {}'.format(i_epoch, validation_acc, validation_f1, validation_auc))

        # save best model
        if validation_acc > max_accuracy:
            max_accuracy = validation_acc

            if not os.path.exists(args.model_output_directory):
                os.mkdir(args.model_output_directory)

            checkpoint_file = 'checkpoint.pth'

            dict_to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }

            output_dir = os.path.join(args.model_output_directory, checkpoint_file)
            torch.save(dict_to_save, output_dir)

    # torch.cuda.empty_cache()
    logger.info('Train done')

def test(args, model, device, data, processor):

    load_file = os.path.join(args.model_output_directory, 'checkpoint.pth')
    checkpoint = torch.load(load_file, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    # clip_params = list(map(id, model.model.parameters()))
    # base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    # optimizer = AdamW([
    #         {'params': base_params},
    #         {'params': model.model.parameters(),'lr': args.clip_lr}
    #         ], lr=args.lr, weight_decay=args.weight_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if 'optimizer' in checkpoint:
        optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
        # print('With optimizer & scheduler!')

    epoch_loss, acc, f1, auc = evaluate(args, model, device, data, processor)

    # test results
    wandb.log({'test_loss': epoch_loss, 'test_acc': acc, 'test_f1': f1, 'test_auc': auc})
    logging.info('test_loss is {}, test_acc is {}, test_f1 is {}, test_auc is {}'.format(epoch_loss, acc, f1, auc))

    # torch.cuda.empty_cache()
    logger.info('Test done')