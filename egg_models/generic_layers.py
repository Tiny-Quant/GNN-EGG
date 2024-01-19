
# %% Dependencies

# torch 
import torch 
import torch.distributions as td 
import torch.nn as nn 
from torch.multiprocessing import Pool  

# %% 
class ContFeatMatrix(nn.Module):
    '''
    Generates a continuous feature matrix based on a linear transformation of 
    a sampled standard random normal vector. 

    returns a continuous and differentiable tensor of size [batch, obs, feats].
    '''
    def __init__(self, batch_size, num_obs, num_feats):
        super(ContFeatMatrix, self).__init__()
        self.batch_size = batch_size
        self.num_obs = num_obs
        self.num_feats = num_feats

        self.linear = nn.Linear(self.num_obs, 
            self.num_obs * self.num_feats)

        # ensures Z will de created on the same device.
        self.register_buffer('Z', torch.randn((self.batch_size, self.num_obs)))

    def forward(self):
        X = self.linear(self.Z)
        X = X.view(self.batch_size, self.num_obs, self.num_feats)
        return X

# %%
class CatFeatVector(nn.Module):
    '''
    Generates a discrete feature vector sampled from a categorical distribution.
    The parameters of the distribution defined to be trainable based on the logLik.

    returns a non-differentiable tensor of size [batch, obs, 1] and a 
    differentiable logLik tensor of size [batch].
    '''
    def __init__(self, batch_size, num_obs, num_cats):
        super(CatFeatVector, self).__init__()
        self.batch_size = batch_size
        self.num_obs = num_obs
        self.num_cats = num_cats

        # self.logits = nn.Parameter(
        #     nn.init.xavier_normal_( # Glorot initialization. 
        #         torch.empty((1, self.num_cats))
        #     )
        # )

        self.probs = nn.Parameter(
            nn.init.uniform_(
                torch.empty((1, self.num_cats)), 
                0.0, 1.0
            )
        )

    def forward(self):
        # dist = td.Categorical(logits=self.logits)
        dist = td.Categorical(probs=torch.softmax(self.probs, dim=0))
        sample = dist.sample(
            (self.batch_size, self.num_obs)
        ) # sample adds an extra dim.
        logLik = dist.log_prob(sample).sum(dim=(1, 2))

        return sample, logLik

# %%
class BinaryMatrix(nn.Module):
    '''
    Generates a matrix where each entry is sampled from an independent 
    Bernoulli distribution (trainable parameters).

    returns a non-differentiable tensor of size [batch, num_rows, num_cols]
    and a tensor containing the logLik of each matrix of size [batch].
    '''
    def __init__(self, batch_size, num_rows, num_cols):
        super(BinaryMatrix, self).__init__()
        self.batch_size = batch_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        # self.logits = nn.Parameter(
        #     nn.init.xavier_normal_( # glorot initialization. 
        #         torch.empty((self.num_rows, self.num_cols))
        #     ).fill_diagonal_(-1e8)
        # )

        self.probs = nn.Parameter(
            nn.init.uniform_(
                torch.empty((self.num_rows, self.num_cols)),
                0.0, 1.0
            ).fill_diagonal_(0.0) # no self-loops.
        )

    def forward(self):
        # dist = td.Bernoulli(logits=self.logits)
        dist = td.Bernoulli(probs=self.probs.clamp(0.0, 1.0))
        sample = dist.sample([self.batch_size])
        logLik = dist.log_prob(sample).sum(dim=(1, 2))
        return sample, logLik
