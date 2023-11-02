import torch

def flatten(list_to_flatten):
    ## flatten text from List[List[Strings]] -> List[Strings]
    lengths = [ len(i) for i in list_to_flatten]
    flattened_list = [item for sublist in list_to_flatten for item in sublist]

    return flattened_list, lengths

def unflatten(flattened_list, lengths):
    pooled_embeddings = []
    start_idx = 0
    for size in lengths:
        end_idx = start_idx + size
        pooled_embedding = torch.mean(flattened_list[start_idx:end_idx, :], dim=0)
        pooled_embeddings.append(pooled_embedding)
        start_idx = end_idx
    
    pooled_tensor = torch.stack(pooled_embeddings, dim=0)
    
    return pooled_tensor

def text_mean_pooling(model_output, attention_mask):
    ## pool embeddings of 'last_hidden_state'
    text_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(text_embeddings.size()).float()
    
    return torch.sum(text_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)