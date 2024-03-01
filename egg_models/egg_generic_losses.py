import torch 
import torch.nn as nn 


# %%

class PredLossBatched(nn.Module):
    def __init__(self, target: torch.Tensor, 
                 explainee: nn.Module,
                 criterion = nn.BCELoss(reduction='none')):

        super(PredLossBatched, self).__init__()
        self.target = target
        self.explainee = explainee
        self.criterion = criterion
    
    def forward(self, batch):
        explainee_pred = torch.softmax(self.explainee(batch), dim=1)
        loss = self.criterion(explainee_pred, 
                              self.target.expand_as(explainee_pred))

        return loss