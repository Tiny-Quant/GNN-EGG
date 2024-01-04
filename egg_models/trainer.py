# %% Dependencies:
import torch 
import torch.nn as nn

from utils.ceograph import NucleiData, clear_iso_nodes
from egg_models.losses import PredLoss, EditLoss

class Trainer:
    def __init__(self, 
                 generator: nn.Module, 
                 explainee: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: callable, 
                 target, obs: list):
        self.generator = generator
        self.explainee = explainee
        self.criterion = criterion
        self.optimizer = optimizer

        self.target = target
        self.obs = obs

    def train_one_epoch(self, lambda_1=1, lambda_2=1, lambda_3=1):

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
        pred_loss = (pred_losses.sum() + 
                     pred_rewards @ -C_x_logLik + 
                     pred_rewards @ -A_logLik) / self.generator.batch_size

        edit_loss_fn = EditLoss(self.obs)
        edit_dists = edit_loss_fn(graph_list) / len(self.obs)
        edit_rewards = 1 / edit_dists + 1 # smoother. 
        edit_loss = (edit_rewards @ -C_x_logLik.repeat(len(self.obs)) + 
                     edit_rewards @ -A_logLik.repeat(len(self.obs)))

        edge_pen = torch.norm(self.generator.AdjacencyMatrix.logits, p=1)

        loss = (lambda_1 * pred_loss +
                lambda_2 * edit_loss + 
                lambda_3 * edge_pen)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_epochs=1):
        for epoch in range(num_epochs):
            self.train_one_epoch()


