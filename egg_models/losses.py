
# %% Dependencies:
import torch 
import torch.nn as nn 

# %%
class PredLoss(nn.Module):
    def __init__(self, target, criterion, explainee):
        super(PredLoss, self).__init__()
        self.target = target
        self.criterion = criterion
        self.explainee = explainee

    def pred_loss_fn(self, example):
        try: 
            explainee_pred = torch.softmax(self.explainee(example), dim=0)

            return self.criterion(explainee_pred, self.target)
        
        except Exception as e:
            return self.criterion(torch.tensor([0.5, 0.5]), self.target)

    def forward(self, examples):
        pred_losses = torch.stack([
            self.pred_loss_fn(example) for example in examples
        ])

        return pred_losses