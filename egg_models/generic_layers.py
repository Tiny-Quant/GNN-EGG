
# %% Dependencies

# torch 
import torch 
import torch.distributions as td 
import torch.nn as nn 
# from torch.multiprocessing import Pool  

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

    def forward(self):
        Z = torch.randn((self.batch_size, self.num_obs), 
                         device=self.linear.weight.device)
        X = self.linear(Z)
        X = X.view(self.batch_size, self.num_obs, self.num_feats)

        return X

# %%
class CatFeatVector(nn.Module):
    '''
    Generates a discrete feature vector sampled from a categorical distribution.
    The parameters of the distribution defined to be trainable based on the 
    logLik.

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
        dist = td.Categorical(probs=torch.softmax(self.probs, dim=1))
        sample = dist.sample(
            (self.batch_size, self.num_obs)
        ) # sample adds an extra dim.
        logLik = dist.log_prob(sample).sum(dim=(1, 2))

        return sample, logLik

# %%
class ConcreteLayer(nn.Module):
    '''
    Generates a one-hot feature matrix with each row sampled from a 
    concrete distribution. The parameters of the distribution 
    defined to be trainable based on the logLik.

    returns a differentiable tensor of size [batch, obs, cats] and a 
    differentiable logLik tensor of size [batch].
    '''
    def __init__(self, batch_size, num_obs, num_cats, temp):
        super(ConcreteLayer, self).__init__()

        self.batch_size = batch_size
        self.num_obs = num_obs
        self.num_cats = num_cats
        self.temp = torch.tensor(temp)

        self.probs = nn.Parameter(
            nn.init.uniform_(
                torch.empty((1, self.num_cats)), 
                0.0, 1.0
            ).squeeze(0)
        )
    
    def forward(self):
        # validate_args=False
        # https://github.com/pyro-ppl/pyro/issues/1640: Suspected Precision Issue 
        # Caused by low temp parameter? 
        dist = td.RelaxedOneHotCategorical(
            self.temp, probs=torch.softmax(self.probs, dim=0), 
            validate_args=False
        )
        sample = dist.rsample(
            (self.batch_size, self.num_obs)
        )
        logLik = dist.log_prob(sample).sum(dim=(1))

        return sample, logLik

# %%
class BinaryMatrix(nn.Module):
    '''
    Generates a matrix where each entry is sampled from independent
    Bernoulli distributions with trainable parameters equal to the sample size.

    returns a non-differentiable tensor of size [batch, num_rows, num_cols]
    and a differentiable tensor containing the joint logLik of each matrix 
    of size [batch].
    '''
    def __init__(self, batch_size, num_rows, num_cols, allow_self_loops=True):
        super(BinaryMatrix, self).__init__()
        self.batch_size = batch_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.allow_self_loops = allow_self_loops

        self.probs = nn.Parameter(
            nn.init.uniform_(
                torch.empty((self.num_rows, self.num_cols)),
                0.0, 1.0
            ).fill_diagonal_(0.0) # no self-loops.
        )

    def forward(self):
        if not self.allow_self_loops: 
            self.probs.fill_diagonal_(0.0)
        dist = td.Bernoulli(probs=self.probs.clamp(0.0, 1.0))

        sample = dist.sample([self.batch_size])
        logLik = dist.log_prob(sample).sum(dim=(1, 2))

        return sample, logLik

# %%
class BinaryConcrete(nn.Module):
    '''
    Generates a matrix where each entry is sampled from an independent 
    Relaxed Bernoulli distributions with trainable parameters equal to 
    the sample size.

    returns a differentiable tensor of size [batch, num_rows, num_cols]
    and a differentiable tensor containing the joint logLik of each matrix 
    of size [batch].
    '''
    def __init__(self, batch_size, num_rows, num_cols, temp, 
                 allow_self_loops):
        super(BinaryConcrete, self).__init__()
        self.batch_size = batch_size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.temp = torch.tensor(temp)
        self.allow_self_loops = allow_self_loops

        self.probs = nn.Parameter(
            nn.init.uniform_(
                torch.empty((self.num_rows, self.num_cols)),
                0.0, 1.0
            )
        )

    def forward(self):
        if not self.allow_self_loops: 
            self.probs.data.fill_diagonal_(0.0)
        self.probs.data.clamp_(0.0, 1.0)
        dist = td.RelaxedBernoulli(self.temp, probs=self.probs)

        sample = dist.rsample([self.batch_size])
        logLik = dist.log_prob(sample).sum(dim=(1, 2))

        return sample, logLik