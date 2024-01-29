# %% Dependencies:
import math 
from abc import abstractmethod
from tqdm import tqdm

import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity

from utils.ceograph import NucleiData, clear_iso_nodes
from egg_models.losses import PredLoss, EditLoss

class Trainer:
    def __init__(self, 
                 generator: nn.Module, 
                 explainee: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: callable, 
                 target, obs: list, 
                 tensorboard_path, checkpoint_path):
        self.generator = generator
        self.explainee = explainee
        self.criterion = criterion
        self.optimizer = optimizer

        self.target = target
        self.obs = obs

        self.tensorboard_path = tensorboard_path
        self.writer = SummaryWriter(self.tensorboard_path)
        self.checkpoint_path = checkpoint_path

    def train_one_epoch(self, lambda_1=1, lambda_2=100, lambda_3=1):

        self.optimizer.zero_grad()
        # X[b, nodes, feat]; C_x[b, nodes]; A[b, 2, edges];
        # E[b, edges, feat]; lik[b]
        X, C_x, A, E, C_x_logLik, A_logLik = self.generator()

        graph_list = [
            NucleiData(X, C_x, A, E) for (X, C_x, A, E) in 
            zip(X.unbind(), C_x.unbind(), A.unbind(), E.unbind())
        ]

        graph_list = [
            clear_iso_nodes(graph) for graph in graph_list
        ]

        # Note: The combining of the loss function is done in the training 
        #       to allow each part of the loss to be logged separately. 
        #       It also allows us to separate the reinforce losses from the 
        #       direct ones.

        pred_loss_fn = PredLoss(self.target, self.criterion, self.explainee)
        pred_losses = pred_loss_fn(graph_list) 
        pred_rewards = 1 / pred_losses + 1e-4 # avoid zero division.
        pred_loss = (pred_losses.sum() * 1e6 + 
                     pred_rewards @ -C_x_logLik + 
                     pred_rewards @ -A_logLik) / self.generator.batch_size

        edit_loss_fn = EditLoss(self.obs)
        device = self.generator.device_param.device
        edit_dists = edit_loss_fn(graph_list).to(device) / len(self.obs)
        edit_rewards = 1 / edit_dists + 1 # smoother. 
        edit_loss = (edit_rewards @ -C_x_logLik.repeat(len(self.obs)) + 
                     edit_rewards @ -A_logLik.repeat(len(self.obs)))

        edge_pen = torch.norm(self.generator.AdjacencyMatrix.probs, p=1)

        loss = (lambda_1 * pred_loss +
                lambda_2 * edit_loss + 
                lambda_3 * edge_pen)

        loss.backward()
        self.optimizer.step()
        
        # Additional metrics.
        with torch.no_grad():
            mean_edit_dist = edit_dists.mean()
            mean_pred_loss = pred_loss.mean()
        
        return [loss.item(), pred_loss.item(), 
                edit_loss.item(), 
                edge_pen.item(), 
                mean_edit_dist.item(), 
                mean_pred_loss.item()]

    def train(self, num_epochs=1, save_every=1, 
              resume_path=None, 
              profile_run=False, profile_dir=""):

        if resume_path is not None:
            last_checkpoint = torch.load(resume_path)
            self.generator.load_state_dict(
                last_checkpoint['generator_state_dict']
            )
            self.optimizer.load_state_dict(
                last_checkpoint['optimizer_state_dict']
            )
            start_epoch = last_checkpoint['epoch'] + 1
        else: 
            start_epoch = 0

        total_epochs = start_epoch + num_epochs 
        for epoch in tqdm(range(start_epoch, total_epochs), 
            desc="Epochs", dynamic_ncols=True):
            
            if profile_run and epoch == math.ceil((start_epoch + total_epochs) / 2):
                with profile( 
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    profile_memory=True, with_stack=True,
                    # ref: https://github.com/pytorch/pytorch/issues/100253
                    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
                ) as prof: 
                    (total_loss, pred_loss, 
                    edit_loss, 
                    edge_pen, 
                    mean_edit_dist, 
                    mean_pred_loss) = self.train_one_epoch()
                with open(profile_dir + "/" + "profile.txt", "a+") as f:
                    print(prof.key_averages().table(sort_by="self_cuda_time_total"), 
                          file=f)

            else: 
                (total_loss, pred_loss, 
                edit_loss, 
                edge_pen, 
                mean_edit_dist, 
                mean_pred_loss) = self.train_one_epoch()

            self.writer.add_scalar("Total Loss", total_loss, epoch)
            self.writer.add_scalar("Prediction Loss", pred_loss, epoch)
            self.writer.add_scalar("Mean Prediction Loss", mean_pred_loss, epoch)
            self.writer.add_scalar("Edit Loss", edit_loss, epoch)
            self.writer.add_scalar("Mean Edit Distance", mean_edit_dist, epoch)
            self.writer.add_scalar("Edge Penalty", edge_pen, epoch)

            if epoch % save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'generator_state_dict': self.generator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'total_loss': total_loss,
                }
                checkpoint_filename = f'{self.checkpoint_path}/checkpoint_epoch_{epoch}.pt'
                torch.save(checkpoint, checkpoint_filename)

class BaseTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                checkpoint_path, save_every=1):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
    
    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError

    def per_epoch_logger(self, result):
        pass
    
    def save_checkpoint(self, epoch, total_epochs):

        if epoch % self.save_every == 0 or epoch == total_epochs - 1: 
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            checkpoint_filename = f'{self.checkpoint_path}/checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_filename)

    def train(self, num_epochs=1, resume_path=None, profile_dir=None):

        if resume_path is not None:
            last_checkpoint = torch.load(resume_path)
            self.model.load_state_dict(
                last_checkpoint['generator_state_dict']
            )
            self.optimizer.load_state_dict(
                last_checkpoint['optimizer_state_dict']
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


