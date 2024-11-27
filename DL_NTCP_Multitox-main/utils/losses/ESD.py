import torch
from torch import nn



class ESD_Loss(nn.Module):
    """
    Expected Squared Difference (ESD) loss as described in the paper ``
    """

    def __init__(self,
                 device: torch.device,
                 reduction: str = 'none') -> None:
        super(ESD_Loss, self).__init__()

        self.device = device

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        confidence1 = torch.sigmoid(logits)
        original_shape = confidence1.shape
        #N1 = len(confidence1)  # number of predictions in this batch
        N1 = torch.numel(confidence1)
        #confidence1 = torch.sigmoid(confidence1)
        confidence1= confidence1.view(N1)
        correct = targets.view(N1)

        val = correct.float() - confidence1   
        val = val.view(1,N1) 
        mask = torch.ones(N1,N1) - torch.eye(N1)
        mask = mask.to(self.device)
        confidence1_matrix = confidence1.expand(N1,N1)
        temp = (confidence1.view(1,N1).T).expand(N1,N1)
        tri = torch.le(confidence1_matrix,temp).float() 
        val_matrix = val.expand(N1,N1)
        x_matrix = torch.mul(val_matrix,tri)*mask
        g_bar = torch.sum(x_matrix, dim = 1)/(N1-1)
        x_matrix_squared = torch.mul(x_matrix, x_matrix)
        S_g = 1/(N1-2) * torch.sum(x_matrix_squared,dim=1) - (N1-1)/(N1-2) * torch.mul(g_bar,g_bar)

        # this is the ESD loss inside the square brackets of equation 10
        ESD_sum = torch.mul(g_bar, g_bar) - S_g/(N1-1)

        ESD_sum = torch.reshape(ESD_sum, original_shape)

        return ESD_sum