import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from models.utils import get_autoencoder_model
from misc import make_config_object

#import config


class M3D_Seg(Dataset):
    """
    A Pytorch Dataset class for the M3D Segmentation dataset.
    In train mode, random crops are taken from the data samples. In test mode, the center crop is taken.
    """
    def __init__(self, data_folder_dir, crop_size = 96, random_cropping=False, masked=False, test_mode=False):
        self.data_folder = data_folder_dir
        self.data_samples = os.listdir(data_folder_dir)
        #self.data = data
        self.crop_size = crop_size
        self.random_cropping = random_cropping
        self.test_mode = test_mode
        self.masked = masked

        if test_mode:
            self.data_samples = self.data_samples[:32]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        # Return a single data sample
        data = np.load(os.path.join(self.data_folder, self.data_samples[index]))
        data = torch.tensor(data, dtype=torch.float32) # turn into tensor and remove the first dimension
        # Get the shape of the data tensor
        data_shape = data.shape

        # if training mode, crop a random 96x96x96 cube from the data tensor
        if self.random_cropping:
            # Generate random coordinates for the starting point of the cube
            start_x = random.randint(0, data_shape[1] - self.crop_size)
            start_y = random.randint(0, data_shape[2] - self.crop_size)
            start_z = random.randint(0, data_shape[3] - self.crop_size)

            # Crop the cube from the data tensor
            #cropped_cube = data[:, start_x:start_x+self.crop_size, start_y:start_y+self.crop_size, start_z:start_z+self.crop_size]
        
        # if test mode, crop the 96x96x96 cube from the center data tensor
        else:
            # Crop the cube from the center of the data tensor
            start_x = (data_shape[1] - self.crop_size) // 2
            start_y = (data_shape[2] - self.crop_size) // 2
            start_z = (data_shape[3] - self.crop_size) // 2

    
        # Crop the cube from the data tensor
        cropped_cube = data[:, start_x:start_x+self.crop_size, start_y:start_y+self.crop_size, start_z:start_z+self.crop_size]

        # if training a masked autoencoder, randomly drop 20% of the blocks in the cube
        if self.masked:
            masked_cube = cropped_cube.clone()

            block_size = 12
            num_blocks_per_axis = 8

            total_blocks = num_blocks_per_axis ** 3

            # randomly select 20% of these blocks to drop
            num_blocks_to_drop = int(0.2 * total_blocks)
            zero_block_indices = torch.randint(0, total_blocks, (num_blocks_to_drop,))

            # convert to 3d coords
            zero_block_indices_3d = torch.stack([zero_block_indices % num_blocks_per_axis,
                                                    (zero_block_indices // num_blocks_per_axis) % num_blocks_per_axis,
                                                    zero_block_indices // (num_blocks_per_axis ** 2)], dim=1)

            zero_block_indices_3d = zero_block_indices_3d * block_size
            #zero_block_indices_3d = zero_block_indices_3d.T

            for coord in zero_block_indices_3d:
                masked_cube[:, coord[0]:coord[0]+block_size, coord[1]:coord[1]+block_size, coord[2]:coord[2]+block_size] = 0

            return masked_cube, cropped_cube
        
        else:
            return cropped_cube, cropped_cube





def train_batch(config, model, train_dataloader, optimizer, loss_function):
    train_loss = 0
    n_batches = len(train_dataloader)
    
    model.train()

    for input, target in tqdm(train_dataloader):
        input = input.to(config.device)
        target = target.to(config.device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * target.shape[0]

    return train_loss / n_batches

def validate_autoencoder(config, model, val_dataloader, loss_function):
    val_loss = 0
    n_batches = len(val_dataloader)
    
    model.eval()

    with torch.no_grad():
        for input, target in tqdm(val_dataloader):
            input = input.to(config.device)
            target = target.to(config.device)
            outputs = model(input)
            loss = loss_function(outputs, target)
            
            val_loss += loss.item() * target.shape[0]

    return val_loss / n_batches





def main(model_name=None, logger=None, override_config=None, TEST_MODE = False):
    

    batch_size = 16 if not TEST_MODE else 4
    num_epochs = 100 if not TEST_MODE else 5
    patience = 10

    input_size = (batch_size,1,96,96,96)
    batch_size, channels, depth, height, width = input_size

    if override_config is None:
        import config as config
        if model_name is not None:
            config.model_name = model_name

        
        config.use_best_config = True
        config = make_config_object(config)

        config.exp_dir = os.path.join(config.root_path, 'experiments', "autoencoder_model", config.model_name)
        config.n_input_channels = channels
    else:
        config = override_config
        #TEST_MODE = config.perform_test_run

    #config.model_name = "resnet_lrelu"
    model = get_autoencoder_model(config, channels, depth, height, width, n_features=0, logger=logger, save_summary=True)

    # Create your dataset
    train_dataset = M3D_Seg('datasets/M3D_Seg/train', crop_size=96, random_cropping=True, masked=True, test_mode=TEST_MODE)
    test_dataset = M3D_Seg('datasets/M3D_Seg/test', crop_size=96, random_cropping=False, masked=True, test_mode=TEST_MODE)


    # Create a DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=8, shuffle=True, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,  num_workers=4, shuffle=False, persistent_workers=True, pin_memory=True)

    
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #dummy_input = torch.randn(*input_shape)
    """
    Model Training
    """

    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    epochs_not_improved = 0

    wandb.login(key="c7f0f65fac8b7178ad7c5859ba6114775b16e694")
    wandb.init(project="test AE TL pipeline", job_type="train")  
    wandb.watch(model, log_freq=100)  
    

    os.makedirs(config.exp_dir, exist_ok=True)

    for epoch in range(num_epochs):  # Number of epochs
        
        train_loss = train_batch(config, model, train_dataloader, optimizer, loss_function)

        val_loss = validate_autoencoder(config, model, test_dataloader, loss_function)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_not_improved = 0

            torch.save(model.state_dict(), os.path.join(config.exp_dir, config.filename_best_model_pth))    
        else:
            epochs_not_improved += 1

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}. \nBest Val Loss: {best_val_loss:.4f}. Epochs not improved: {epochs_not_improved}')
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if epochs_not_improved >= patience:
            break

    wandb.finish()


    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.title(f"{config.model_name} Autoencoder Losses")
    plt.legend()
    plt.savefig(os.path.join(config.exp_dir, f"{config.model_name}_losses.png"))
    plt.close()


if __name__ == '__main__':  
    from data_preproc.data_preproc_functions import Logger
    logger=Logger()   

    folder_name = "Masked_Autoencoder_0"

    #models = ["dcnn_pooling","resnet_lrelu", "ViT", "resnext_lrelu"]
    models = ["dcnn_pooling","resnet_lrelu", "ViT"]
    models = ["resnet_lrelu"]


    for m in models:
        main(folder_name=folder_name, model_name=m, logger=logger)

    # if not (os.name == 'nt'):
    #     torch.multiprocessing.set_start_method("spawn")
    #     torch.multiprocessing.set_sharing_strategy("file_system")