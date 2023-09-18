import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        def forward(self, sims, logit_scale):
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * sims

            text2video_log_sm = F.log_softmax(logits, dim=1)
            text2video_neg_ce = torch.diagonal(text2video_log_sm)
            text2video_loss = -text2video_neg_ce.mean()

            video2text_log_sm = F.log_softmax(logits, dim=0)
            video2test_neg_ce = torch.diagonal(video2text_log_sm)
            video2text_loss = -video2test_neg_ce.mean()

            return (text2video_loss + video2text_loss) / 2
        
class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        return Loss() #config.loss == 'clip'