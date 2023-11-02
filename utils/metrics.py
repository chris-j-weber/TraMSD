import torch
import numpy as np
from sklearn import metrics
from utils.utils import flatten

def evaluate(args, model, device, criterion, data, text_processor, vision_processor):
    
    prob = []
    y_pred = []
    targets = []
    running_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(data):
            if args.model in ['fusion', 'cross_attention']:
                videos, text, label = batch
                frames = torch.flatten(videos, start_dim=0, end_dim=1).to(device)
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
            running_loss += loss.item()

    loss = running_loss / len(data)

    targets = torch.tensor(targets).to(device)
    y_pred = np.argmax(np.array(prob), axis=-1)
    acc = metrics.accuracy_score(targets.cpu(), y_pred)
    auc = metrics.roc_auc_score(targets.cpu(), np.array(prob)[:, 1])
    f1 = metrics.f1_score(targets.cpu(), y_pred, pos_label=1)

    return loss, acc, auc, f1