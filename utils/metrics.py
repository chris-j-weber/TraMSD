#import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from data.dataset import Mustard

def accuracy_eval(args, model, device, data, processor, macro=False, pre=None, mode='test'):
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
            
            score, logits_fused, logits_text, targets = model(inputs,labels=labels)

            loss_fuse = loss_function(logits_fused, labels)
            loss_text = loss_function(logits_text, labels)
            #loss_image = self.loss_fct(logits_image, labels)
            
            #loss = loss_fuse + loss_text + loss_image
            loss = loss_fuse + loss_text

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
    #if mode == 'test':
        #wandb.log({'test_loss': sum_loss/sum_step})
    #else:
        #wandb.log({'val_loss': sum_loss/sum_step})
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

def similarity_matrix_training(text_embeddings, video_embeddings_pooled):
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    video_embeddings_pooled = video_embeddings_pooled / video_embeddings_pooled.norm(dim=-1, keepdim=True)

    text_embeddings = text_embeddings.unsqueeze(1)
    video_embeddings_pooled = video_embeddings_pooled.permute(1, 2, 0)
    sims = torch.bmm(text_embeddings, video_embeddings_pooled).squeeze(1)
    return sims

def similarity_matrix_inference(text_embeddings, video_embeddings_pooled):
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    video_embeddings_pooled = video_embeddings_pooled / video_embeddings_pooled.norm(dim=-1, keepdim=True)

    num_videos, max_text, embedding_dimension = text_embeddings.shape

    video_embeddings_pooled = video_embeddings_pooled.permute(1, 2, 3, 0)
    video_embeddings_pooled = video_embeddings_pooled.view(num_videos*max_text, embedding_dimension, num_videos)

    text_embeddings = text_embeddings.unsqueeze(2)
    text_embeddings = text_embeddings.view(num_videos*max_text, 1, embedding_dimension)

    sims = torch.bmm(text_embeddings, video_embeddings_pooled)
    sims = sims.view(num_videos, max_text, 1, num_videos).squeeze(2)

    return sims

def generate_embeddings(text_embd, vid_embd, all_vids):
    text_embd_id = {}

    for idx, v_id in enumerate(all_vids):
        if v_id in text_embd_id:
            text_embd_id[v_id].append(text_embd[idx])
        else:
            text_embd_id[v_id] = [text_embd[idx]]

    for v_id in text_embd_id:
        text_embd_id[v_id] = torch.stack(text_embd_id[v_id])

    text_embd_id = pad_and_stack_dict_to_tensor(text_embd_id, text_embd_id.keys(), text_embd.shape[-1])

    vid_embd_id = []
    for i in range(vid_embd.shape[0]):
        vid_embd_id.append({})
        for idx, v_id in enumerate(all_vids):
            if v_id in vid_embd_id[i]:
                vid_embd_id[i][v_id].append(vid_embd[i, idx, :])
            else:
                vid_embd_id[i][v_id] = [vid_embd[i, idx, :]]

    for i in range(len(vid_embd_id)):
        for v_id in vid_embd_id[i]:
            vid_embd_id[i][v_id] = torch.stack(vid_embd_id[i][v_id])
            vid_embd_id[i] = pad_and_stack_dict_to_tensor(vid_embd_id[i], vid_embd_id[i].keys(), vid_embd.shape[-1])

    vid_embd_id = torch.stack(vid_embd_id)
    return text_embd_id, vid_embd_id

def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                        float("-inf"), device = input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input

def text2video_metric(sims):
    stacked_sims = sims.permute(1,0,2)
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort2 = torch.argsort(sims_sort, dim=-1, descending=False)
    ranks = torch.flatten(torch.diagonal(sims_sort2, dim1=1, dim2=2))

    valid_check = torch.flatten(torch.diagonal(sims, dim1=0, dim2=2))
    mask = ~torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())

def video2text_metric(sims):
    sims[sims!=sims] = float('-inf')
    sims, _ = torch.max(sims, dim=1)
    sims = sims.t()

    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort2 = torch.argsort(sims_sort, dim=-1, descending=False)

    ranks = torch.diag(sims_sort2).numpy()
    return compute_metrics(ranks)

def compute_metrics(ranks):
    metrics =  {}
    metrics['meanR'] = np.mean(ranks)+1
    metrics['medianR'] = np.median(ranks)+1
    return metrics