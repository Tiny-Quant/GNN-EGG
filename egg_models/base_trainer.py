
import math 
from abc import abstractmethod
from tqdm import tqdm

import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity

class BaseTrainer:
    """
    Generic trainer class for pytorch models. 

    Handles tensorboard, checkpoint saving, and the outside epoch 
    training logic. Also contains an option for profile the training 
    performance and bottlenecks.
    """
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 tensorboard_path: str, 
                 checkpoint_path: str, 
                 save_every=1):

        self.model = model
        self.optimizer = optimizer

        self.tensorboard_path = tensorboard_path
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
        self.writer = SummaryWriter(self.tensorboard_path)

        self.scaler = torch.cuda.amp.GradScaler(init_scale=3_000)

    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError

    def per_epoch_logger(self, result: dict, epoch: int):

        for key in result.keys():
            self.writer.add_scalar(key, result[key], epoch)
    
    def save_checkpoint(self, epoch, total_epochs):

        if epoch % self.save_every == 0 or epoch == total_epochs - 1: 
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(), 
            }
            checkpoint_filename = f'{self.checkpoint_path}/checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_filename)

    def train(self, num_epochs=1, resume_path=None, profile_dir=None):

        if resume_path is not None:
            last_checkpoint = torch.load(resume_path)
            self.model.load_state_dict(
                last_checkpoint['model_state_dict']
            )
            self.optimizer.load_state_dict(
                last_checkpoint['optimizer_state_dict']
            )
            self.scaler.load_state_dict(
                last_checkpoint['scaler']
            )
            start_epoch = last_checkpoint['epoch'] + 1
        else: 
            start_epoch = 0

        total_epochs = start_epoch + num_epochs 
        for epoch in tqdm(range(start_epoch, total_epochs), desc="Training", leave=True):
            
            if profile_dir is not None and epoch == math.ceil((start_epoch + total_epochs) / 2):
                with profile( 
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    profile_memory=True, with_stack=True,
                    # ref: https://github.com/pytorch/pytorch/issues/100253
                    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
                ) as prof: 
                    result = self.train_one_epoch()
                with open(profile_dir + "/" + "profile.txt", "a+") as f:
                    print(prof.key_averages().table(sort_by="self_cuda_time_total"), 
                          file=f)

            else: 
                result = self.train_one_epoch()

            self.per_epoch_logger(result, epoch)

            self.save_checkpoint(epoch, total_epochs)


