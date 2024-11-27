import torch
from torch import nn





class SB_ECE_Loss(nn.Module):
    """
    Soft-Binning Expected Calibration Error (SB-ECE) 
    """
    def __init__(self,
                 n_bins: int = 10,
                 T: float = 0.01,
                 device: torch.device = torch.device('cpu'),
                 reduction: str = 'none') -> None:
        super(SB_ECE_Loss, self).__init__()

        self.n_bins = n_bins
        self.T = T
        self.device = device
        self.reduction = reduction
        
        # find the centers of each bin
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)[:-1]  #  (remove the last element as torch.linspace `does` include the end point (np.linspace does not))
        bin_width = ((bin_boundaries[1] - bin_boundaries[0])/2) 
        self.bin_centers = bin_boundaries + bin_width       

        self.columnwise_softmax = torch.nn.Softmax(dim=0)

        self.eps = 1e-8
    

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        N = len(logits)  # get the number of predictions in this batch
        m = self.n_bins
        confidences_all = torch.sigmoid(logits + self.eps).T
        targets_all = targets.T

        bin_center = self.bin_centers

        loss_template = torch.zeros_like(confidences_all, device=self.device, dtype=torch.float32)

        for idx, (confidences, targets) in enumerate(zip(confidences_all, targets_all)):
            #print(idx)
            #print(confidences)
            N = len(confidences)  # get the number of predictions in this batch
            r_hat = confidences.view(1, N)
            r_hat_matrix = r_hat.expand(m, N)
            epsilon_hat = bin_center.view(m, 1)
            epsilon_hat_matrix = epsilon_hat.expand(m, N).to(self.device)

            g = -(r_hat_matrix - epsilon_hat_matrix)**2 / self.T
            g_matrix = self.columnwise_softmax(g)

            correctness_vector = targets.float().view(1, N)
            correctness_matrix = correctness_vector.expand(m, N)
            A_matrix = correctness_matrix * g_matrix

            S = torch.sum(g_matrix, dim=1)
            S_check = torch.eq(S, torch.zeros(m).to(self.device)).float()
            A_hat = torch.div(torch.sum(A_matrix, dim=1), S + S_check + self.eps).view(m, 1)
            A_hat_matrix = A_hat.expand(m, N)
            temp_matrix = ((A_hat_matrix - r_hat_matrix)**2) * g_matrix

            loss =  (1/N * torch.sum(temp_matrix) + self.eps)**(1/2)

            loss_template[idx] = loss
        
        loss_template = torch.clamp(loss_template, min=0, max=1)
        return loss_template.T



        




class SB_ECE_Loss_OLD(nn.Module):
    """
    Soft-Binning Expected Calibration Error (SB-ECE) 
    """
    def __init__(self,
                 n_bins: int = 10,
                 T: float = 0.01,
                 device: torch.device = torch.device('cpu'),
                 reduction: str = 'none') -> None:
        super(SB_ECE_Loss, self).__init__()

        self.n_bins = n_bins
        self.T = T
        self.device = device
        self.reduction = reduction
        
        # find the centers of each bin
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)[:-1]  #  (remove the last element as torch.linspace `does` include the end point (np.linspace does not))
        bin_width = ((bin_boundaries[1] - bin_boundaries[0])/2) 
        self.bin_centers = bin_boundaries + bin_width       

        self.columnwise_softmax = torch.nn.Softmax(dim=0)
    

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor ) -> torch.Tensor:
        confidences_all = torch.sigmoid(logits)
        targets_all = targets
        #N = len(confidences)  # get the number of predictions in this batch
        #orig_shape = confidences_all.shape
        
        m = self.n_bins

        loss_template = torch.zeros_like(confidences_all)

        for idx, (confidences, targets) in enumerate(zip(confidences_all, targets_all)):
            N = torch.numel(confidences)

            bin_centers_matrix = self.bin_centers.view(m,1) .expand(m,N).to(self.device)  # bin centers as a matrix (each row contains a bin center, but the dimension is the same as the predictions matrix)
        
            pred_confidences_matrix = confidences.view(1,N).expand(m,N)  # predictions as a tensor of size m x N (each prediction is a column, and there is one row per bin)
            # compute the soft ECE bin probabilities
            
            g = -(pred_confidences_matrix-bin_centers_matrix)**2/ self.T  # soft bin function
            g_matrix = self.columnwise_softmax(g)        

            true_labels_matrix = targets.float().view(1,N).expand(m,N)
            Accuracy_matrix = true_labels_matrix * g_matrix
            bin_accuracies = torch.sum(Accuracy_matrix, dim = 1)  # accuracy of each bin

            S = torch.sum(g_matrix, dim = 1) # size of each bin
            S_check = torch.eq(S, torch.zeros(m).to(self.device)).float()  
            A_hat = torch.div(bin_accuracies, S+S_check).view(m,1)  # mean accuracy of all bins
            A_hat_matrix = A_hat.expand(m,N)
            temp_matrix = ((A_hat_matrix - pred_confidences_matrix)**2)*g_matrix

            temp2 = torch.sum(temp_matrix, dim = 0)

            loss_template[idx] = temp2
            
        return loss_template

        return (1/N*torch.sum(temp_matrix))**(1/2)
        #loss = torch.reshape(temp_matrix, orig_shape)
        #return loss