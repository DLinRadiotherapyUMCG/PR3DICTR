import time
from typing import Dict, Optional, Any

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device

import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


ray.shutdown()
ray.init(configure_logging=False)

def evaluate(step, width, height, activation):
    time.sleep(0.1)
    assert torch.cuda.is_available()

    device = get_device()
    assert device == torch.device("cuda:0")
    
    device  = get_device()
    model = MyModel(10, int(width), 2)
    model.to(device)
    activation_boost = 10 if activation=="relu" else 0
    return (0.1 + width * step / 100) ** (-1) + height * 0.1 + activation_boost


def objective(config):
    for step in range(config["steps"]):
        score = evaluate(step, config["width"], config["height"], config["activation"])
        train.report({"iterations": step, "mean_loss": score})


search_space = {
    "steps": 100,
    "width": tune.uniform(0, 20),
    "height": tune.uniform(-100, 100),
    "activation": tune.choice(["relu", "tanh"]),
}

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=4)
num_samples = 10


tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=algo,
        num_samples=num_samples,
    ),
    scaling_config=ScalingConfig(
        num_workers=1,
        use_gpu=True
    ),
    param_space=search_space
)
results = tuner.fit()