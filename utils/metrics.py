import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from data.dataset import Mustard

def accuracy_eval(args, model, device, data, processor, macro=False, pre=None, mode='test'):
    if args.model == 'fusion':
        data_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=False, num_workers=args.num_workers)
    else:
        # data_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=False, num_workers=args.num_workers, drop_last=True)
        data_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=False, num_workers=args.num_workers)
    
    n_correct, n_total = 0, 0
    all_targets, all_outputs = None, None

    model.eval()
    sum_loss = 0.
    sum_step = 0

    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            text_list, image_list, label_list, id_list = t_batch
            inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.text_max_len, return_tensors="pt").to(device)
            labels = torch.tensor(label_list).to(device)
            
            targets = labels
            # loss, t_outputs = model(inputs,labels=labels)
            if args.model == 'fusion':
                ##### model_V1 #####
                score, logits_fused, logits_text, targets = model(inputs,labels=labels)

                loss_fuse = loss_function(logits_fused, labels)
                loss_text = loss_function(logits_text, labels)
                loss = loss_fuse + loss_text
                ########
            else:
                ##### model_V2 #####
                logits, score, labels = model(inputs,labels=labels)

                logits = logits.view(-1, 2)
                labels = labels.repeat(4)

                loss_output = loss_function(logits, labels)
                loss = loss_output
                ########

            sum_loss += loss.item()
            sum_step += 1
  
            outputs = torch.argmax(score, -1)

            n_correct += (outputs == targets).sum().item()
            n_total += len(outputs)

            if all_targets is None:
                all_targets = targets
                all_outputs = outputs
            else:
                all_targets = torch.cat((all_targets, targets), dim=0)
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
    # if mode == 'test':
    #     wandb.log({'test_loss': sum_loss/sum_step})
    # else:
    #     wandb.log({'val_loss': sum_loss/sum_step})
    if pre != None:
        with open(pre,'w',encoding='utf-8') as fout:
            predict = all_outputs.cpu().numpy().tolist()
            label = all_targets.cpu().numpy().tolist()
            for x,y,z in zip(predict,label):
                fout.write(str(x) + str(y) +z+ '\n')
    if not macro:   
        acc = n_correct / n_total
        f1 = metrics.f1_score(all_targets.cpu(), all_outputs.cpu())
        precision =  metrics.precision_score(all_targets.cpu(),all_outputs.cpu())
        recall = metrics.recall_score(all_targets.cpu(),all_outputs.cpu())
    else:
        acc = n_correct / n_total
        f1 = metrics.f1_score(all_targets.cpu(), all_outputs.cpu(), labels=[0, 1],average='macro')
        precision =  metrics.precision_score(all_targets.cpu(),all_outputs.cpu(), labels=[0, 1],average='macro')
        recall = metrics.recall_score(all_targets.cpu(),all_outputs.cpu(), labels=[0, 1],average='macro')
    return acc, f1, precision, recall