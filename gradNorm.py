



class GradNorm:
    def __init__(self, model, norm_type=2):
        self.model = model
        self.norm_type = norm_type

    def grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(self.norm_type)
            total_norm += param_norm.item() ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)
        return total_norm
    
    