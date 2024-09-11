import logging

import torch
import wandb
from torch.utils.data import DataLoader

from src.constants import DEVICE
from src.utils.loss_func.get_loss_function import get_loss_function
from src.models.tools.get_model import get_model
from src.utils.optimizer.get_optimizer import get_optimizer
from src.utils.scheduler.get_scheduler import get_scheduler
from src.evaluation.calculate_auc import calculate_auc


def train(config, train_data, val_data, metadata):
    """
    Train the model.
    :param config:
    :param train_data:
    :param val_data:
    :param metadata:
    :return: Model
    """
    # Get the data loaders
    train_loader = DataLoader(train_data, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['training']['batch_size'], shuffle=False)

    # Get the model
    logging.info('Getting model')
    model = get_model(config, metadata)
    model.to(device=DEVICE)
    # wandb.watch(model, log_freq=100)

    # Get loss function, optimizer, and scheduler
    loss_function = get_loss_function(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # Initialize the best model and lowest validation loss
    best_model = None
    highest_auc = - float('inf')
    patience_counter = 0

    # Training loop
    logging.info('Starting training loop')
    for epoch in range(config['training']['max_epochs']):

        logging.info(f'Starting epoch {epoch}')
        model.train()

        total_loss = 0.0
        total_auc = 0.0
        num_batches = 0
        num_auc_batches = 0

        for i, batch in enumerate(train_loader):
            logging.debug(f'Batch {i} of epoch {epoch}')

            optimizer.zero_grad(set_to_none=True)
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            # Make predictions
            outputs = model(x=inputs, clinical_features=clinical_features)

            # Calculate loss
            loss = loss_function(outputs, targets)
            loss.backward()

            # Calculate AUC
            # Sigmoid activation function is applied to the outputs
            sigmoid = torch.nn.Sigmoid()
            auc = calculate_auc(sigmoid(outputs), targets)

            # Log loss and AUC
            total_loss += loss.item()
            if auc is not None:
                total_auc += auc
                num_auc_batches += 1

            # Update model weights
            optimizer.step()

            # Step the scheduler
            scheduler.step(epoch + (i + 1) / len(train_loader))

            num_batches += 1

        # Log epoch loss and AUC
        avg_loss = total_loss / num_batches
        avg_auc = total_auc / num_auc_batches
        logging.info(f'Epoch loss: {avg_loss}')
        logging.info(f'Epoch AUC: {avg_auc}')

        # Perform validation
        if epoch % config['training']['validation_interval'] == 0:
            val_loss, val_auc = validate(loss_function, model, val_loader)
            # wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc, 'val/loss': val_loss, 'val/auc': val_auc})

            # Check if this model has the lowest validation loss
            if val_auc > highest_auc:
                logging.info(f'New highest AUC: {val_auc}')
                highest_auc = val_auc
                best_model = model.state_dict()  # Save the model state
                patience_counter = 0 # Reset patience counter
            else:
                patience_counter += 1 # Increment patience counter

            # Check if patience has been exhausted
            if patience_counter >= config['training']['patience']:
                logging.info('Patience exhausted, stopping training')
                break
        # else:
        #     wandb.log({'train/loss': avg_loss, 'train/auc': avg_auc})

    # Log highest AUC
    # wandb.log({'train/highest_auc': highest_auc})
    # logging.info('Finished training')

    model.load_state_dict(best_model)

    return model


def validate(loss_function, model, val_loader):
    model.eval()
    total_loss = 0.0
    total_auc = 0.0
    num_batches = 0
    num_auc_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.debug(f'Validation batch {i}')
            inputs, clinical_features, targets = move_batch_to_device(batch, DEVICE)

            outputs = model(x=inputs, clinical_features=clinical_features)
            loss = loss_function(outputs, targets)
            sigmoid = torch.nn.Sigmoid()
            auc = calculate_auc(sigmoid(outputs), targets)

            total_loss += loss.item()
            if auc is not None:
                total_auc += auc
                num_auc_batches += 1

            num_batches += 1

    model.train()

    avg_loss = total_loss / num_batches
    avg_auc = total_auc / num_auc_batches

    logging.debug(f'Validation loss: {avg_loss}')
    logging.debug(f'Validation AUC: {avg_auc}')

    return avg_loss, avg_auc


def move_batch_to_device(batch, device):
    inputs, clinical_features, targets = batch
    inputs = inputs.to(device=device)
    clinical_features = clinical_features.to(device=device)
    targets = targets.to(device=device)
    return inputs, clinical_features, targets
