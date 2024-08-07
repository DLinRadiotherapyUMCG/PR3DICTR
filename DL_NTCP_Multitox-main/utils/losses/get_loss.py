"""
Loss functions and classes for training the pytorch models
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss


from utils.losses.dice import DiceLoss
from utils.losses.F1 import F1_Loss
from utils.losses.L1 import L1Loss
from utils.losses.ranking import RankingLoss
from utils.losses.softAUC import SoftAUCLoss
from utils.losses.custom import CustomLoss
from utils.losses.focal import FocalLoss
from utils.losses.asymmetric import AsymmetricLossOptimized
from utils.losses.hill import Hill
from utils.losses.SB_ECE import SB_ECE_Loss
from utils.losses.ESD import ESD_Loss
from utils.losses.MLLSC import MLLSC
from utils.losses.SPLC import SPLC


class WeightedLossWrapper(nn.Module):
    def __init__(self, loss_function, device, label_weights=None, label_pos_weights=None):
        super(WeightedLossWrapper, self).__init__()
        self.loss_function = loss_function
        self.device = device
        # weights = how much each classes' weight is weighted relative to the other classes
        if label_weights is None: 
            self.label_weights=None
        else: 
            self.label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device) # / sum(label_weights) # normalised to 1
        # pos_weight = how much each positive instance is weighted, relative to the negative class
        if label_pos_weights is None: 
            self.label_pos_weights=None
        else: 
            self.label_pos_weights = torch.tensor(label_pos_weights, dtype=torch.float32, device=device) # / sum(label_pos_weights) # normalised to 1
        
        

    def forward(self, input, target):
        # get the loss as normal (using the loss function)
        batch_loss = self.loss_function(input, target)

        # now apply the per-class weights to the loss
        if self.label_weights is not None:
            batch_loss *= self.label_weights

        # apply the pos_weight to the losses
        if self.label_pos_weights is not None:
            pos_mask = torch.isin(target, torch.tensor([1], device=self.device)) # only the positive examples
            pos_weight_multiplier = pos_mask * self.label_pos_weights         # per-class pos_weight (2 for first, 3 for second)
            pos_weight_multiplier[pos_weight_multiplier==0] = 1
            batch_loss *= pos_weight_multiplier

        # masking out the missing labels is done in the main code 
        return batch_loss



class AuxillaryLossWrapper(nn.Module):
    def __init__(self,
                 config,
                 main_loss_function, 
                 ) -> None:
        super(AuxillaryLossWrapper, self).__init__()

        self.main_loss_function = main_loss_function
        
        if config.auxiliary_loss_func_name == "SB-ECE":
            self.aux_loss_function = SB_ECE_Loss(n_bins=config.SB_ECE_n_bins, T=config.SB_ECE_temperature, device=config.device, reduction='none')
        elif config.auxiliary_loss_func_name == "ESD":
            self.aux_loss_function = ESD_Loss(device=config.device, reduction='none')

    def forward(self, logits, targets):
        main_loss = self.main_loss_function(logits, targets)
        aux_loss = self.aux_loss_function(logits, targets)

        #print(main_loss, aux_loss)

        return main_loss + aux_loss




def class_ratio_counter(config, train_dict):
    num_endpoints = len(config.endpoint_list)
    num_patients =  np.zeros(num_endpoints) # counts the number of patients that have either 0 or 1 for each endpoint (i.e. that are not missing data)
    class_counts = np.zeros(num_endpoints)  # counts the number of patients that are equal to 1 (i.e. have the toxicity)
    for patient in train_dict:
        labels = np.array(patient['label_list'])
        data_present = labels>=0
        num_patients += data_present
        labels[labels<0]=0
        class_counts += labels

    pos_weights = num_patients/(2*class_counts)
    inv_class_counts = num_patients - class_counts

    neg_weights = num_patients/(2*inv_class_counts)

    # print("POS WEIGHTS", class_counts)
    # print("TOTAL NUM  ", num_patients)
  
    return inv_class_counts/class_counts  # weights the positive samples against the number of negative samples


def get_loss_function(config, train_dict, val_dict):
    """
    CrossEntropyLoss: softmax is computed as part of the loss. In other words, the model outputs should be logits,
        i.e. the model output should NOT be softmax'd.
    BCEWithLogitsLoss: sigmoid is computed as part of the loss. In other words, the model outputs should be logits,
        i.e. the model output should NOT be sigmoid'd.

    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
    Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

    Args:
        config: (class object) configuration parameters
        train_dict: (dict) dictionary containing the training data (for setting the label weights)
        val_dict: (dict) dictionary containing the validation data
    Returns:
        loss function (class object)

    """
    loss_function_name = config.loss_function_name
    label_weights = config.label_weights
    label_pos_weights = config.label_pos_weights
    reduction = config.loss_reduction

    if loss_function_name == 'cross_entropy':
        loss_function = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                  label_smoothing=config.label_smoothing)
    elif loss_function_name == 'bce':
        if label_pos_weights == True:
            label_pos_weights = class_ratio_counter(config=config, train_dict=train_dict)
            label_pos_weights = torch.tensor(label_pos_weights, dtype=torch.float32, device=config.device)
            # print(" BCE POS WEIGHTS")
            # print(label_pos_weights)
        else:
            label_pos_weights = None
            # print(" BCE POS WEIGHTS")
            # print(label_pos_weights)
        loss_function = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=label_pos_weights)
        # TODO: replace weight with pos_weight to deal with imbalanced classes
    #elif loss_function_name == 'dice':
    #    loss_function = DiceLoss()
    elif loss_function_name == 'f1':
        loss_function = F1_Loss().to(config.device)
    #elif loss_function_name == 'ranking':
    #    loss_function = RankingLoss(reduction=reduction)
    elif loss_function_name == 'soft_auc':
        loss_function = SoftAUCLoss()
    elif loss_function_name == 'mse':
        loss_function = MSELoss(reduction="none")
    #elif loss_function_name == 'custom':
    #    loss_function = CustomLoss(config=config, reduction=reduction)
    elif loss_function_name == 'focal':
        loss_function = FocalLoss(reduction=reduction, alpha=config.alpha, gamma=config.gamma, margin=config.margin)  # TODO: add the parameters alpha and gamma
    elif loss_function_name == 'asymmetric':
        loss_function = AsymmetricLossOptimized(gamma_neg=config.gamma_neg, gamma_pos=config.gamma_pos)  # TODO: add the parameters
    elif loss_function_name == 'hill':
        loss_function = Hill(lamb=config.lamb, margin=config.margin, gamma=config.gamma, reduction=reduction)
    elif loss_function_name.lower() == 'splc':
        loss_function = SPLC(tau=config.tau, margin=config.margin, gamma=config.gamma, reduction=reduction)
    elif loss_function_name.lower() == 'mllsc':
        loss_function = MLLSC(variant=config.mllsc_variant, tau_pos=config.tau_pos, tau_neg=config.tau_neg, margin=config.margin, gamma=config.gamma, reduction=reduction)

    else:
        raise ValueError('Invalid loss_function_name: {}.'.format(loss_function_name))
    
    if config.auxiliary_loss_func_name != None:
        loss_function = AuxillaryLossWrapper(config=config, main_loss_function=loss_function)
        
        loss_function.pos_weight = label_pos_weights  # needed to check if BCE is using label_pos_weights

    # if (label_weights is not None) or (label_pos_weights is not None):
    #     # weight the losses per class
    #     loss_function = WeightedLossWrapper(loss_function, device=config.device, label_weights=label_weights, label_pos_weights=None)
    return loss_function



def calc_mixup_loss(predictions, targets, mixup_y_true_indexes, loss_function, mixup_lambda):
    #batch_loss = loss_function(predictions, targets)
    y_a = targets
    y_b = targets[mixup_y_true_indexes]

    #print(predictions.shape)
    #print(targets.shape)
    predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)


    #predictions = torch.reshape(predictions, y_a.shape).to(predictions.dtype)
    #predictions = torch.reshape(predictions, y_b.shape).to(predictions.dtype)

    y_a_loss = loss_function(predictions, y_a)
    y_b_loss = loss_function(predictions, y_b)

    batch_loss = (mixup_lambda * y_a_loss) + ((1 - mixup_lambda) * y_b_loss)

    return batch_loss




def calculate_loss(cfg, outputs_dict, labels_dict, valid_endpoints_as_tensor, loss_function, mixup_y_true_indexes=None, mixup_lambda=None):

    predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32) # transposed! so that num columns = num toxicities
    targets = torch.stack(list(labels_dict.values()), dim=1).type(torch.float32)

    # compute loss
    if (cfg.perform_mixup == True) and (mixup_y_true_indexes != None) and (mixup_lambda != None):
        batch_loss = calc_mixup_loss(predictions, targets, mixup_y_true_indexes, loss_function, mixup_lambda)
    else:
        predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)
        batch_loss = loss_function(predictions, targets)
 
    
    per_toxicity_loss_dict = dict()
    axis = 0
    if not cfg.loss_function_name == 'mse':
        mask = (targets >= valid_endpoints_as_tensor[0]) & (targets <= valid_endpoints_as_tensor[1])
        batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
        batch_loss *= mask

        # find the mean loss of the non-missing labels
        batch_loss_mean = batch_loss.sum() / mask.sum()
        # find the loss per endpoint (the column-wise mean of the batch_loss tensor)
        per_toxicity_loss = batch_loss.sum(axis=axis) / mask.sum(axis=axis)
    else:
        batch_loss_mean = batch_loss.mean()

        per_toxicity_loss = batch_loss.mean(axis=axis)

    for idx, endpoint in enumerate(cfg.endpoint_list):
        value = per_toxicity_loss[idx].item()
        per_toxicity_loss_dict[endpoint] = value if (not math.isnan(value)) else 0
    

    # also get the non-weighted loss
    # if we are not weighting the loss, then the unweighted_batch_loss_mean = batch_loss_mean, and unweighted_per_toxicity_loss_dict = per_toxicity_loss_dict
    unweighted_per_toxicity_loss_dict = dict()
    if cfg.label_pos_weights:
    #if loss_function.pos_weight:
        pos_weights = loss_function.pos_weight

        unweighting_factor = torch.where(targets != 1, torch.ones_like(targets), pos_weights.expand(batch_loss.shape))
        unweighted_batch_loss = batch_loss / unweighting_factor
        unweighted_batch_loss_mean = unweighted_batch_loss.sum() / mask.sum()

        unweighted_per_toxicity_loss = unweighted_batch_loss.sum(axis=axis) / mask.sum(axis=axis)
        for idx, endpoint in enumerate(cfg.endpoint_list):
            value = unweighted_per_toxicity_loss[idx].item()
            unweighted_per_toxicity_loss_dict[endpoint] = value if (not math.isnan(value)) else 0
        
    else:
        unweighted_batch_loss_mean = batch_loss_mean
        for idx, endpoint in enumerate(cfg.endpoint_list):
            unweighted_per_toxicity_loss_dict[endpoint] = per_toxicity_loss_dict[endpoint]

    #print(batch_loss_mean)
    batch_loss_mean = torch.clamp(batch_loss_mean, min=0, max=10000) 

    return batch_loss_mean, per_toxicity_loss_dict, unweighted_batch_loss_mean, unweighted_per_toxicity_loss_dict

