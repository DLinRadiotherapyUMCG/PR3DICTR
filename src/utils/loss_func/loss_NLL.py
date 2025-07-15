import torch
import torch.nn as nn

from src.constants import MISSING_DATA_VALUE



class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class
        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.
        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss



class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        #self.L2_reg = 0.00005
        #self.reg = Regularization(order=2, weight_decay=self.L2_reg)

        self.missing_endpoints_as_tensor = torch.tensor([MISSING_DATA_VALUE])
        self.eps = 0.1  # 1e-8
        self.sigmoid = torch.nn.Sigmoid()

    def set_endpoint_list(self, events_endpoint_list):
        """
        Set the list of endpoints for the loss function.
        :param endpoint_list: List of endpoints (not used in this implementation)
        """
        self.events_endpoint_list = events_endpoint_list
        self.num_tasks = len(events_endpoint_list)
        

    def forward(self, risk_pred, y, e):
        """
        :param risk_pred: (batch_size, num_tasks)
        :param y: (batch_size, num_tasks)
        :param e: (batch_size, num_tasks)
        """
        # risk_pred = risk_pred.cuda()
        # y = y.cuda()
        # e = e.cuda()

        
        batch_size, num_tasks = risk_pred.shape
        assert num_tasks == self.num_tasks, "Mismatch in number of tasks"

        # NOTE: TEMP
        risk_pred = self.sigmoid(risk_pred)  # Apply sigmoid activation to risk predictions

        zero_tensor = torch.tensor(0.0, device=risk_pred.device, dtype=risk_pred.dtype)
        #total_loss = zero_tensor
        loss_dict = {event : zero_tensor for event in self.events_endpoint_list}

        for k, event_name in enumerate(self.events_endpoint_list):

            # Identify valid (non-missing) samples for this task
            valid_mask = (e[:, k] != self.missing_endpoints_as_tensor[0])

            if valid_mask.sum() < 2:
                #neg_log_loss = zero_tensor
                loss_dict[event_name] = zero_tensor

            else:

                y_k = y[valid_mask, k].unsqueeze(1)  # (B, 1)
                e_k = e[valid_mask, k]
                risk_k = risk_pred[valid_mask, k].unsqueeze(1)  # (B, 1)

                valid_batch_size = y_k.shape[0]

                # Masking: patients at risk
                mask = torch.ones(valid_batch_size, valid_batch_size, device=y_k.device) # .cuda()
                mask[(y_k.T - y_k) > 0] = zero_tensor

                log_loss = torch.exp(risk_k) * mask
                log_loss = torch.sum(log_loss, dim=0) # / (torch.sum(mask, dim=0) + self.eps)
                log_loss = torch.log(log_loss).reshape(-1, 1)
                neg_log_loss = -torch.sum((risk_k - log_loss) * e_k.unsqueeze(1)) / (torch.sum(e_k) + self.eps)

                #total_loss += neg_log_loss
                loss_dict[event_name] = neg_log_loss / valid_batch_size

        return loss_dict
    


