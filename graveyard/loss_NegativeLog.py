import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class Regularization(object):
    def __init(self,order, weight_decay):
        ''' The initialization of Regularization class
        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calcualtes regularization (self.order) loss for model
        :param model: (roch.nn.Module object)
        :return reg_loss: (Torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w,p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss
    
class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super(NegativeLogLikelihood,self).__init__()
        self.L2_reg = 0.00005
        self.reg = Regularization(order = 2, weight_decay = config['training']['optimizer']['weight_decay'])

    def forward(self, risk_pred, y, e, model):
        ''' Calculating the negative log likelihood
        param risk_pred: Prediction of risk for given event
        param y: label for time of event
        param e: label for name of event
        param model: 
        '''
        risk_pred = risk_pred.cuda()
        y = y.cuda()
        e = e.cuda()

        mask = torch.ones(y.shape[0],y.shape[0]).cuda()
        y = y.unsqueeze(1).cuda()
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0)/ (torch.sum(mask,dim=0)+0.1)
        log_loss = torch.log(log_loss).reshape(-1,1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e)/(torch.sum(e)+0.1)
        l2_norm = self.reg(model)
        return neg_log_loss + l2_norm, neg_log_loss, l2_norm