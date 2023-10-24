import os
import torch
import logging
import wandb
import torch.nn as nn
from data.dataset import Mustard
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils.metrics import accuracy_eval

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args, train_data, val_data, test_data, model, processor, device):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.model == 'fusion':
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=True, num_workers=args.num_workers)
    else:
        # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=True, num_workers=args.num_workers, drop_last=True)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=True, num_workers=args.num_workers)
    total_steps = int(len(train_loader) * args.num_train_epoches)
    model.to(device)
    
    clip_params = list(map(id, model.model.parameters()))
    base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    optimizer = AdamW([
            {'params': base_params},
            {'params': model.model.parameters(),'lr': args.clip_lr}
            ], lr=args.lr, weight_decay=args.weight_decay)
    
    loss_function = nn.CrossEntropyLoss()

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)

    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epoches), desc='Epoch', disable=False):
        sum_loss = 0.
        sum_step = 0.

        iter_bar = tqdm(train_loader, desc='Iter (loss=X.XXX)', disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.text_max_len, return_tensors='pt').to(device)
            labels = torch.tensor(label_list).to(device)

            res = model(inputs,labels=labels)
            if args.model == 'fusion':
                ###### model_V1 #####
                logits_fused = res[1]
                logits_text = res[2]
                labels = res[3]

                loss_fuse = loss_function(logits_fused, labels)
                loss_text = loss_function(logits_text, labels)
                loss = loss_fuse + loss_text
                ########
            else:
                ##### model_V2 #####
                logits_output = res[0]
                # video_features_pooled = res[1]
                score = res[1]
                labels = res[2]
                logits_output = logits_output.view(-1, 2)
                labels = labels.repeat(4)
                loss_output = loss_function(logits_output, labels)
                loss = loss_output
                ########

            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()

        # wandb.log({'train_loss': sum_loss/sum_step})
        val_acc, val_f1 ,val_precision,val_recall = accuracy_eval(args, model, device, val_data, processor, mode='dev')
        #wandb.log({'val_acc': val_acc, 'val_f1': val_f1, 'val_precision': val_precision, 'val_recall': val_recall})
        # wandb.log({'val_acc': val_acc, 'val_f1': val_f1})
        #logging.info("i_epoch is {}, val_acc is {}, val_f1 is {}, val_precision is {}, val_recall is {}".format(i_epoch, val_acc, val_f1, val_precision, val_recall))
        logging.info("i_epoch is {}, val_acc is {}, val_f1 is {}".format(i_epoch, val_acc, val_f1))

        if val_acc > max_acc:
            max_acc = val_acc

            path_to_save = os.path.join(args.output_dir, 'mmt')
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

            test_acc, test_f1,test_precision,test_recall = accuracy_eval(args, model, device, test_data, processor,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = accuracy_eval(args, model, device, test_data, processor, mode='test')
            #wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1, 'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_, 'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            # wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1, 'micro_test_f1': test_f1_})
            #logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, micro_test_f1 is {}".format(i_epoch, test_acc, test_f1, test_f1_))

        torch.cuda.empty_cache()
    logger.info('Train done')
