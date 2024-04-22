# %%
from tqdm import tqdm

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import torch.distributions as td

# import torch_geometric as pyg
from torch_geometric.data import Data, Batch

import pygmtools as pygm
from pygmtools.utils import dense_to_sparse
pygm.set_backend('pytorch')

from egg_models.generic_layers import ContFeatMatrix, ConcreteLayer, BinaryConcrete
from egg_models.trainer import BaseTrainer
# from egg_models.losses import PredLossBatched, MatchingLoss 
from utils.ceograph import (
    assign_edge_type, NucleiData, load_model, clear_iso_nodes, 
    get_nuclei_batch
)

# %%
class EggSoft(nn.Module):
    def __init__(self, node_size, cont_node_feat, node_types, cont_edge_feat,
                 temp_1=1.0, temp_2=1.0,
                 batch_size=1): 
        super(EggSoft, self).__init__()
        self.node_size = node_size
        self.cont_node_feat = cont_node_feat       
        self.node_types = node_types
        self.cont_edge_feat = cont_edge_feat
        self.temp_1 = temp_1
        self.temp_2 = temp_2
        self.batch_size = batch_size

        # Layers 
        self.ContNodeFeats = ContFeatMatrix(
            self.batch_size, self.node_size, self.cont_node_feat, 
        )
        self.DisNodeFeats = ConcreteLayer(
            self.batch_size, self.node_size, self.node_types, 
            self.temp_1
        )
        self.AdjacencyMatrix = BinaryConcrete(
            self.batch_size, self.node_size, self.node_size, 
            self.temp_2
        )
        self.ContEdgeFeats = ContFeatMatrix(
            self.batch_size, self.node_size**2, self.cont_edge_feat
        )

        self.device_param = nn.Parameter(torch.empty(0))
    
    def forward(self):
        X = self.ContNodeFeats()
        
        C_x, C_x_logLik = self.DisNodeFeats()
        #C_x = C_x.to(self.device_param.device)
        # print(C_x.device, self.device_param.device)

        A, A_logLik = self.AdjacencyMatrix()
        #A.to(self.device_param.device)

        E = self.ContEdgeFeats()

        # No Diff - for pred_loss: 
        with torch.no_grad():
            C_x_hard = torch.argmax(C_x, dim=-1) + 1
            A_hard = (A >= 0.5)
            edge_indices_hard = dense_to_sparse(A_hard)[0].transpose(1, 2)
            # TODO: further abstraction needed for benchmark datasets. 
            edge_types_hard = assign_edge_type(
                edge_indices_hard, C_x_hard
            ).to(self.device_param.device)
            E_hard = E.narrow(dim=1, start=0, length=edge_indices_hard.shape[2])
            edge_attr_hard = torch.cat((edge_types_hard, E_hard), dim=-1)

        # Diff - for discriminators:
        # print(X.device, C_x.device)
        node_matrix = torch.cat((X, C_x), dim=-1)
        edge_indices, edge_weights, _ = dense_to_sparse(A)
        edge_indices = edge_indices.transpose(1, 2)
        edge_attr = assign_edge_type(
            edge_indices, C_x_hard
        ).to(self.device_param.device)
        edge_attr = torch.cat((edge_attr, E, edge_weights), dim=-1)

        # Wrap 
        # nuclei_list = [clear_iso_nodes(NucleiData(X, C_x, A, E))
        #                 for (X, C_x, A, E) in zip(
        #                     X.unbind(), C_x_hard.unbind(), edge_indices_hard.unbind(), 
        #                     edge_attr_hard.unbind()
        #               )]
        #nuclei_batch = Batch().from_data_list(nuclei_batch)

        # data_list = [Data(X, A, E) 
        #               for (X, A, E) in zip(
        #                 node_matrix.unbind(), edge_indices.unbind(), 
        #                 edge_attr.unbind()
        #             )]
        # data_batch = Batch().from_data_list(data_batch)

        results_dict = {'X_shared': X, 
                        'C_x_hard': C_x_hard, 
                        'A_hard': edge_indices_hard, 
                        'E_hard': edge_attr_hard, 
                        'node_matrix_soft': node_matrix, 
                        'A_soft': edge_indices, 
                        'E_soft': edge_attr, 
                        'C_x_logLik': C_x_logLik, 
                        'A_logLik': A_logLik
        }

        return results_dict

# %%
class EggSoftTrainer(BaseTrainer): 
    def __init__(self,
                model: EggSoft, 
                explainee: nn.Module, 
                obs_loader: DataLoader, 
                target: torch.Tensor, 
                pred_loss: nn.Module, 
                struct_loss: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                tensorboard_path, 
                checkpoint_path, save_every=1, 
                samples_per_param=1, 
                reinforce_pred=False, reinforce_struct=False, 
                lambda_1=1, lambda_2=1, lambda_3=1):

        super().__init__(model, optimizer, checkpoint_path, save_every)

        self.explainee = explainee
        self.obs_loader = obs_loader
        self.target = target

        self.pred_loss = pred_loss 
        self.reinforce_pred = reinforce_pred
        self.struct_loss = struct_loss
        self.reinforce_struct = reinforce_struct
        self.samples_per_param = samples_per_param
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.tensorboard_path = tensorboard_path
        self.checkpoint_path = checkpoint_path
        self.save_every = save_every
        self.writer = SummaryWriter(tensorboard_path)

    def train_one_epoch(self):
        self.optimizer.zero_grad()
        running_total_loss = 0
        running_pred_loss = 0
        running_match_loss = 0
        running_edge_pen = 0
        for i, obs_batch in enumerate(tqdm(self.obs_loader, desc="Batching", leave=False)):

            generated = self.model()

            nuclei_batch = get_nuclei_batch(generated['X_shared'],
                                            generated['C_x_hard'], 
                                            generated['A_hard'], 
                                            generated['E_hard'])
            pred_loss = self.pred_loss(nuclei_batch).mean(dim=1)

            # print(pred_loss.shape)
            # print(generated['C_x_logLik'].shape)

            if self.reinforce_pred:
                pred_loss = (pred_loss.mean() + 
                             (1 / pred_loss) @ 
                             (-generated['C_x_logLik'] +  -generated['A_logLik']))

            else: 
                pred_loss = pred_loss.mean()

            match_loss = self.struct_loss(obs_batch, 
                                          generated['node_matrix_soft'], 
                                          generated['A_soft'], 
                                          generated['E_soft'])

            if self.reinforce_struct:
                match_loss = (match_loss.mean() + 
                              (1 / match_loss) @ 
                              (-generated['C_x_logLik'] +  -generated['A_logLik']))
            
            else: 
                match_loss = match_loss.mean()
            
            edge_pen = (torch.norm(self.model.AdjacencyMatrix.probs, p=2) + 
                        nn.functional.softplus(
                            self.model.AdjacencyMatrix.probs.sum() - 
                            (self.model.node_size)) ** 2)

            total_loss = (self.lambda_1 * pred_loss +
                          self.lambda_2 * match_loss + 
                          self.lambda_3 * edge_pen)

            total_loss.backward()

            if (i+1) % self.samples_per_param == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_total_loss += total_loss.item()
            running_pred_loss += self.lambda_1 * pred_loss.item()
            running_match_loss += self.lambda_2 * match_loss.item()
            running_edge_pen += self.lambda_3 * edge_pen.item()

        result = {
            'total_loss': running_total_loss / len(self.obs_loader),
            'pred_loss': running_pred_loss / len(self.obs_loader),
            'match_loss': running_match_loss / len(self.obs_loader),
            'edge_pen': running_edge_pen / len(self.obs_loader)
        }
        
        return result

    def per_epoch_logger(self, result, epoch):
        self.writer.add_scalar("Total Loss", result['total_loss'], epoch)
        self.writer.add_scalar("Prediction Loss", result['pred_loss'], epoch)
        self.writer.add_scalar("Matching Loss", result['match_loss'], epoch)
        self.writer.add_scalar("Edge Penalty", result['edge_pen'], epoch)
