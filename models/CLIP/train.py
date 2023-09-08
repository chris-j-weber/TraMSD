import os
import torch
import logging
#import wandb
import numpy as np
from data.dataset import Mustard
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn import metrics
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(args, train_data, val_data, test_data, model, processor, device):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=True, num_workers=args.num_workers)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)
    
    clip_params = list(map(id, model.model.parameters()))
    base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    optimizer = AdamW([
            {'params': base_params},
            {'params': model.model.parameters(),'lr': args.clip_lr}
            ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps), num_training_steps=total_steps)

    max_acc = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc='Epoch', disable=False):
        sum_loss, sum_step = 0.
        #sum_step = 0

        iter_bar = tqdm(train_loader, desc='Iter (loss=X.XXX)', disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.text_max_len, return_tensors='pt').to(device)
            labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()

        #wandb.log({'train_loss': sum_loss/sum_step})
        val_acc, val_f1 ,val_precision,val_recall = evaluate_acc_f1(args, model, device, val_data, processor, mode='dev')
        #wandb.log({'val_acc': val_acc, 'val_f1': val_f1, 'val_precision': val_precision, 'val_recall': val_recall})
        logging.info("i_epoch is {}, val_acc is {}, val_f1 is {}, val_precision is {}, val_recall is {}".format(i_epoch, val_acc, val_f1, val_precision, val_recall))

        if val_acc > max_acc:
            max_acc = val_acc

            path_to_save = os.path.join(args.output_dir, 'mmt')
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

            test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(args, model, device, test_data, processor,macro = True, mode='test')
            _, test_f1_,test_precision_,test_recall_ = evaluate_acc_f1(args, model, device, test_data, processor, mode='test')
            #wandb.log({'test_acc': test_acc, 'macro_test_f1': test_f1,
            #         'macro_test_precision': test_precision,'macro_test_recall': test_recall, 'micro_test_f1': test_f1_,
            #         'micro_test_precision': test_precision_,'micro_test_recall': test_recall_})
            logging.info("i_epoch is {}, test_acc is {}, macro_test_f1 is {}, macro_test_precision is {}, macro_test_recall is {}, micro_test_f1 is {}, micro_test_precision is {}, micro_test_recall is {}".format(i_epoch, test_acc, test_f1, test_precision, test_recall, test_f1_, test_precision_, test_recall_))

        torch.cuda.empty_cache()
    logger.info('Train done')

def evaluate_acc_f1(args, model, device, data, processor, macro=False, pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=False, num_workers=args.num_workers)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.text_max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)
                
                t_targets = labels
                loss, t_outputs = model(inputs,labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        #if mode == 'test':
            #wandb.log({'test_loss': sum_loss/sum_step})
        #else:
            #wandb.log({'val_loss': sum_loss/sum_step})
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:   
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu())
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu())
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu())
        else:
            acc = n_correct / n_total
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1],average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), labels=[0, 1],average='macro')
        return acc, f1, precision, recall
