import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from config import Config
from utils.metrics import similarity_matrix_training, similarity_matrix_inference, generate_embeddings

class Trainer():
    def __init__(self, model, loss, metrics, optimizer, config: Config, train_data, valid_data, tokenizer, lr_scheduler=None, writer=None):
        self.device = self._prepare_device()

        
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config
        self.writer = writer
        self.start_epoch = 1
        self.step = 0
        self.num_epoch = config.num_epochs
        self.train_losses = deque(maxlen=100)
        self.val_losses = deque(maxlen=100)
        self.checkpoint_dir = config.model_path
        self.log_step = config.log_step
        self.evals_per_epoch = config.evals_per_epoch
        self.train_data = train_data
        self.valid_data = valid_data
        self.tokenizer = tokenizer
        self.lr_scheduler = lr_scheduler

        self.pooling = config.pooling
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data)
        eval_steps = np.linspace(0, num_steps-1, self.evals_per_epoch+1, dtype=int)[1:]

        for batch_idx, data in enumerate(self.train_data):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            data['video'] = data['video'].to(self.device)

            text_embd, video_embd = self.model(data)
            #output = similarity_matrix_training(text_embd, video_embd, self.pooling)
            output = similarity_matrix_training(text_embd, video_embd)

            loss = self.loss(output, self.model.clip.logit_scale)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            torch.clamp_(self.model.clip.logit_scale.data, max=np.log(100))

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)

            total_loss += loss.detach().item()

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss.detach().item()))

            if batch_idx in eval_steps:
                val_res = self._valid_epoch_step(epoch, batch_idx, num_steps-1)
                self.model.train()

                if val_res['R1-window'] > self.best_window:
                    self.best_window = val_res['R1-window']
                    self._save_checkpoint(epoch, save_best=True)

                if val_res['R1'] > self.best:
                    self.best = val_res['R1']

                print(" Current Best Window Average R@1 is {}".format(self.best_window))
                print(" Current Best R@1 is {}\n\n".format(self.best))

        res = {
            'loss_train':  total_loss / num_steps
        }

        return res

    def _valid_epoch_step(self, epoch, step, num_steps):
        self.model.eval()
        total_val_loss = 0.0
        text_embed_arr = []
        vid_embed_arr = []
        all_vid_ids = []

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_data_loader)):
                if self.tokenizer is not None:
                    data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
                if isinstance(data['text'], torch.Tensor):
                    data['text'] = data['text'].to(self.device)
                else:
                    data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

                data['video'] = data['video'].to(self.device)
                
                text_embed, vid_embed, vid_embed_pooled = self.model(data, return_all_frames=True)
                text_embed_arr.append(text_embed.cpu())
                vid_embed_arr.append(vid_embed.cpu())
                sims_batch = similarity_matrix_training(text_embed, vid_embed_pooled)

                curr_loss = self.loss(sims_batch, self.model.clip.logit_scale)
                total_val_loss += curr_loss.item()

                for v_id in data['video_id']:
                    all_vid_ids.append(v_id)

            text_embeds = torch.cat(text_embed_arr)
            vid_embeds = torch.cat(vid_embed_arr)

            # Since we have all pairs, remove duplicate videos when there's multiple captions per video
            vid_embeds_per_video_id = {}
            for idx, v_id in enumerate(all_vid_ids):
                if v_id not in vid_embeds_per_video_id:
                    vid_embeds_per_video_id[v_id] = vid_embeds[idx]
            
            vid_embeds = torch.stack([vid_embeds_per_video_id[v_id] for v_id in vid_embeds_per_video_id])
             
            # Pool frames for inference once we have all texts and videos
            self.model.pool_frames.cpu()
            vid_embeds_pooled = self.model.pool_frames(text_embeds, vid_embeds)
            self.model.pool_frames.cuda()

            text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeddings(text_embeds, 
                    vid_embeds_pooled, all_vid_ids)
            
            sims = similarity_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)

            total_val_loss = total_val_loss / len(self.valid_data_loader)

            metrics = self.metrics
            res = metrics(sims)

            # Compute window metrics
            for m in res:
                self.window_metric[m].append(res[m])

            # Compute average of window metrics
            for m in self.window_metric:
                res[m + "-window"] = np.mean(self.window_metric[m])

            print(f"-----Val Epoch: {epoch}, dl: {step}/{num_steps}-----\n",
                  f"R@1: {res['R1']} (window: {res['R1-window']})\n", 
                  f"R@5: {res['R5']} (window: {res['R5-window']})\n", 
                  f"R@10: {res['R10']} (window: {res['R10-window']})\n",
                  f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
                  f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
                  f"Loss: {total_val_loss}")
            
            res['loss_val'] =  total_val_loss

            if self.writer is not None:
                for m in res:
                    self.writer.add_scalar(f'val/{m}', res[m], self.global_step)

            return res

    def train(self):
        for epoch in range(self.start_epoch, self.num_epoch + 1):
            result = self._train_epoch(epoch)
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch, save_best=False)

    def validate(self):
        self._valid_epoch_step(0, 0, 0)

    def _prepare_device(self):
        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda' if use_gpu else 'cpu')
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        return device

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            print('saving best model')
        else:
            model_path = os.path.join(self.checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save(state, model_path)
            print('saving model: {}'.format(model_path))

    def load_checkpoint(self, model_name):
        checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        print('loading checkpoint: {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('loaded checkpoint: {}'.format(checkpoint_path))