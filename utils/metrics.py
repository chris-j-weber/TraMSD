import numpy as np
import torch
import torch.nn.functional as F

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