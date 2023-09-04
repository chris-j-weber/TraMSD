import os
import shutil
import argparse
from abc import ABC

class Config(ABC):
    def __init__(self):
        args = self.parse_args()

        self.dataset_name = args.dataset_name
        self.videos_path = args.videos_path
        self.number_frames = args.number_frames
        self.video_sample_type = args.video_sample_type
        self.input_resolution = args.input_resolution

        self.experiment_name = args.experiment_name
        self.model_path = args.model_path
        self.output_dir = args.output_dir
        self.save_every = args.save_every
        self.log_step = args.log_step
        self.evals_per_epoch = args.evals_per_epoch
        self.load_epoch = args.load_epoch
        self.metric = args.metric

        self.embedding_dimension = args.embedding_dimension
        self.loss = args.loss
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion

        self.pooling = args.pooling
        self.k = args.k
        self.attention_temperature = args.attention_temperature
        self.num_mha_heads = args.num_mha_heads
        self.transformer_dropout = args.transformer_dropout

        self.num_workers = args.num_workers
        self.seed = args.seed
        self.no_tensorboard = args.no_tensorboard
        self.tb_log_dir = args.tb_log_dir

        self.dev_batch_size = args.dev_batch_size
        self.device = args.device
        self.layers = args.layers
        self.text_size = args.text_size
        self.image_size = args.image_size
        self.optimizer = args.optimizer
        self.max_grad_norm = args.max_grad_norm
        self.learning_rate = args.learning_rate


    def parse_args(self):
        description = "mmt"
        parser = argparse.ArgumentParser(description=description)

        parser.add_argument('--dataset_name', type=str, default='mustard')
        parser.add_argument('--videos_path', type=str, default='videos')
        parser.add_argument('--number_frames', type=int, default=12)
        parser.add_argument('--video_sample_type', type=str, default='uniform')
        parser.add_argument('--input_resolution', type=int, default=224)

        parser.add_argument('--experiment_name', type=str, required=True, default='mustard')
        #parser.add_argument('--model_path', type=str, required=True)
        parser.add_argument('--output_dir', type=str, default='./output')
        parser.add_argument('--save_every', type=int, default=1)
        parser.add_argument('--log_step', type=int, default=10)
        parser.add_argument('--evals_per_epoch', type=int, default=10)
        parser.add_argument('--load_epoch', type=int, help='load epoch or -1 to load best model')
        parser.add_argument('--metric', type=str, default='cosine')

        parser.add_argument('--embedding_dimension', type=int, default=512)
        parser.add_argument('--loss', type=str, default='cosine')
        parser.add_argument('--lr', type=float, default=1e-4)

        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
    
    def mkdirp(p):
        if not os.path.exists(p):
            os.makedirs(p)

    def deletedir(p):
        if os.path.exists(p):
            shutil.rmtree(p)
