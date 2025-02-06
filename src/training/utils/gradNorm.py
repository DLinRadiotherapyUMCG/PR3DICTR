import torch
import wandb 

from src.hyper_opt.WandB_functions import WandB_log


class GradNorm(torch.nn.Module):
    def __init__(self, config, model, alpha=1, lr=0.001, WandB_is_enabled=False):
        super(GradNorm, self).__init__()
        self.layer = self.determine_gradnorm_layer(config, model)
        self.alpha = alpha
        self.lr = lr
        self.iters = 0

        self.WandB_is_enabled = WandB_is_enabled

        #self.setup()

        self.log_weights = []
        self.log_loss = []

    def setup(self, loss):
        # to be run on the first iteration
        self.weights = torch.ones_like(loss)
        self.weights = torch.nn.Parameter(self.weights)
        self.T = self.weights.sum().detach()
        self.optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        self.l0 = loss.detach()
        
    def determine_gradnorm_layer(self, config, model):
        # if there are shared linear layers, use the last one of those as the gradnorm layer
        if len(config['model']['linear_units']) > 0:
            #self.gradNorm_layer = self.output_head.linear_layers.shared_fc_layers
            if "transrp" in config['model']['model_name'].lower():
                for layer in model.output_head.linear_layers.shared_fc_layers:
                    if isinstance(layer, (torch.nn.Linear, torch.nn.LazyLinear)):
                        last_linear_layer = layer
            else:
                for layer in model.output_head.shared_fc_layers:
                    if isinstance(layer, (torch.nn.Linear, torch.nn.LazyLinear)):
                        last_linear_layer = layer
                    #last_name = name
            gradNorm_layer = last_linear_layer
                
        else:
            if "transrp" in config['model']['model_name'].lower():
                last_attention_layer_name = f"Attention Block {config['model']['TransRP']['vit_depth'] - 1}"
                gradNorm_layer = getattr(model.output_head.transformer.layers, last_attention_layer_name)[1] # get the last attention block
            else:
                # TODO: a better method for determining the last layer in a normal CNN without any shared linear layers?
                gradNorm_layer = model.encoder   # if no linear layers are present, use the whole image encoder as the layer for GradNorm (hard to define the last layer)

        return gradNorm_layer
            

    def log_to_WandB(self, data):
        if self.WandB_is_enabled:
            #wandb.log(data = data)
            pass

    def step(self, loss):
        if self.iters == 0:
            self.setup(loss)

        weights = self.weights
        
        # compute the weighted loss
        weighted_loss = weights @ loss
        # backward pass for weigthted task loss
        weighted_loss.backward(retain_graph=True)
        # compute the L2 norm of the gradients for each task
        gw = []
        for i in range(len(loss)):
            dl = torch.autograd.grad(weights[i]*loss[i], self.layer.parameters(), retain_graph=True, create_graph=True)[0]
            gw.append(torch.norm(dl))

        gw = torch.stack(gw)
        # compute loss ratio per task
        loss_ratio = loss.detach() / self.l0
        # compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()
        # compute the average gradient norm
        gw_avg = gw.mean().detach()
        # compute the GradNorm loss
        constant = (gw_avg * rt ** self.alpha).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        # clear gradients of weights
        self.optimizer.zero_grad()
        # backward pass for GradNorm
        gradnorm_loss.backward()
        # log weights and loss
        self.log_weights.append(weights.detach().cpu().numpy().copy())
        self.log_loss.append(loss_ratio.detach().cpu().numpy().copy())
        # update gradNorm weights
        self.optimizer.step()
        # renormalize weights
        weights = (weights / weights.sum() * self.T).detach()
        self.weights = torch.nn.Parameter(weights)
        self.optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        self.iters += 1 

        self.log_to_WandB({
            'GradNorm_loss': gradnorm_loss.item(), 
            'GradNorm_weights': self.weights.detach().cpu().numpy().copy(), 
            'GradNorm_loss_ratio': loss_ratio.detach().cpu().numpy().copy(),
            'GradNorm_iteration': self.iters
            })

        return weighted_loss