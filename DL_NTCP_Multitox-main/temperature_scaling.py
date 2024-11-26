from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from scipy.special import expit, logit
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn, optim



"""
Classes: 
1. TemperatureScaler
2. PlattScaler
3. IsotonicRegressionScaler
4. BetaCalibrationScaler
"""

class TemperatureScaler():
    def __init__(self):
        self.lr = LogisticRegression()
    def fit(self, logits, labels):
        # using gradient descent, fits the temperature parameter to the logits and true labels

        logits = logits.float()
        labels = labels.float()

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        nll_criterion = nn.BCEWithLogitsLoss()
        #ece_criterion = ECE_Criterion().cuda()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)


        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits, temperature), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        self.temperature = temperature

        #print(' temperature: ', self.temperature.item())



    def predict_proba(self, logits):  
        # rescales the logits, and applies sigmoid

        return torch.sigmoid(self.temperature_scale(logits, self.temperature).detach())   # have to detatch the tensor, to remove the gradient that comes with fitting the temperature

    
    def temperature_scale(self, logits, temperature):
        # only scales the logits (does not apply sigmoid)
        temperature = temperature.expand(logits.size(0))

        return logits / temperature
    








# 2. Platt scaling
# logistic regression is performed on the logits
class PlattScaler():
    def __init__(self):
        self.lr = LogisticRegression()
    def fit(self, logits, labels):
        #self.X = logits
        #self.y = y
        self.lr.fit(logits.reshape(-1, 1), labels)
    def predict_proba(self, logits):
        return torch.tensor(self.lr.predict_proba(logits.reshape(-1, 1))[:, 1])




# 3. Isotonic regression (for scaling the probabilities)
# isotonic regression is performed on the logits
class IsotonicRegressionScaler():
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds='clip') # must clip out of bounds, or else this will return NaNs
    def fit(self, logits, labels):
        #self.X = X
        #self.y = y
        self.ir.fit(logits, labels)
    def predict_proba(self, logits):
        return torch.tensor(self.ir.transform(logits))




# 4. Beta calibration
# uses dirchelet distribution (s?) to calibrate the probabilities
class BetaCalibration():
    def __init__(self):
        pass
    def fit(self, logits, labels):
        self.logits = logits
        self.labels = labels
        initial_params = torch.tensor([0.5, 0.5])
        res = minimize(self._neg_log_likelihood, initial_params, method='L-BFGS-B')
        #print(res)
        self.alpha_, self.beta_ = res.x
        #print(res.x)

    def _neg_log_likelihood(self, params):
        # compute the negative log likelihood of the beta distribution. Uses the logits and true labels

        alpha, beta = params
        probs = torch.sigmoid(alpha * self.logits + beta)
        
        return -torch.sum(self.labels * torch.log(probs) + (1 - self.labels) * torch.log(1 - probs))
    
    def predict_proba(self, logits):
        return torch.sigmoid(self.alpha_ * logits + self.beta_)