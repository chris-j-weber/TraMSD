import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from data.dataset import Mustard

def evaluate(args, model, device, data, processor):

    model.eval()
    
    criterion = nn.CrossEntropyLoss()

    data_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=Mustard.collate_func, shuffle=False, num_workers=args.num_workers)
    
    prob = []
    y_pred = []
    running_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            text, videos, label, id_list = batch
            frames = torch.flatten(videos, start_dim=0, end_dim=1)
            inputs = processor(text=text, images=frames, padding=True, truncation=True, return_tensors='pt').to(device)
            target = torch.tensor(label).to(device)

            pred = model(inputs)
            prob.extend(torch.nn.functional.softmax(pred, dim=-1).detach().cpu())

            loss = criterion(pred, target)
            running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    y_pred = np.argmax(np.array(prob), axis=-1)
    acc = metrics.accuracy_score(target.cpu(), y_pred)
    auc = metrics.roc_auc_score(target.cpu(), np.array(prob)[:, 1])
    f1 = metrics.f1_score(target.cpu(), y_pred, pos_label=1)

    return epoch_loss, y_pred, acc, auc, f1