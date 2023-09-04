import os
import torch
import random
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from config import Config
from data.data_handler import DataHandler
# import loss, optimizer, trainer
from transformers import CLIPTokenizer
from models.transformer import Transformer
from utils.metrics import text2video_metric, video2text_metric
from utils.optimization import get_cosine_schedule_with_warmup, AdamW
from utils.loss import LossFactory

def main():
    config = Config()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(config.seed)
        random.seed(config.seed)

    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', TOKENIZERS_PARALLELISM=False)

    train_data = DataHandler.get_data_loader(config, 'train')
    val_data = DataHandler.get_data_loader(config, 'val')
    test_data = DataHandler.get_data_loader(config, 'test')

    model = Transformer(config, tokenizer)

    if config.metrics == 'text2video':
        metrics = text2video_metric
    elif config.metrics == 'video2text':
        metrics = video2text_metric
    else:
        raise NotImplemented
    
    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]

    optimizer_grouped_parameters = [
        {"params": noclip_params, "lr": config.no_lr},
        {"params": clip_params, "lr": config.lr}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=config.weight_decay)
    num_training_steps = len(train_data) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, metrics, optimizer, 
                      config=config,
                      train_data_loader=train_data,
                      valid_data_loader=val_data,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer)
    trainer.train()
    

if __name__ == "__main__":
    main()