import torch
from torch import nn

from src.constants import MISSING_DATA_VALUE

from src.utils.loss_func.loss_NLL import NegativeLogLikelihood





class MultiLabel_Loss(nn.Module):
    """
    A class representing the loss function for multi toxicity prediction.
    accepts a loss function (BCE, ASL, focal, etc.) and applies it to the output and labels dictionaries (i.e. it applies a loss function to a multi-tox problem)
    """
    def __init__(self, config, binary_loss_function, LabelTypesManager) -> None:
        super(MultiLabel_Loss, self).__init__()

        self.missing_endpoints_as_tensor = torch.tensor([MISSING_DATA_VALUE])   # this is the value that is used to indicate missing endpoints in the dataset
        self.config = config
        self.endpoints_list = config['columns']['labels']

        self.LabelTypesManager = LabelTypesManager

        
        # set the loss functions
        self.binary_loss_function = binary_loss_function

        self.events_loss_function = NegativeLogLikelihood()
        # set the endpoint list for the events loss function
        self.events_loss_function.set_endpoint_list(self.LabelTypesManager.endpoint_type_groups_names['Event'])


    

    def forward(self, outputs_dict, labels_dict):
        
        # STEP 1: FLATTEN ALL OF THE PREDICITONS
        predictions = torch.stack(list(outputs_dict.values()), dim=1).type(torch.float32).squeeze(-1) # transposed! so that num columns = num toxicities

        # STEP 2: FLATTEN ALL OF THE TARGETS
        targets = labels_dict.squeeze(1)

        """
        BINARY ENDPOINTS (e.g. NTCP)
        """

        # STEP 3: GET ONLY THE BINARY ENDPOINTS
        binary_preds = predictions[:, self.LabelTypesManager.binary_predictions_indicies]
        binary_targets = targets[:, self.LabelTypesManager.binary_targets_indicies]

        # STEP 4: CALCULATE THE LOSS FOR BINARY ENDPOINTS        
        loss_dict_binary_endpoint = self.forward_binary_loss_function(binary_preds, binary_targets, endpoint_list=self.LabelTypesManager.endpoint_type_groups_names['Binary'])

        """
        EVENT ENDPOINTS (e.g. OS, LRC, DMFS, etc.)
        """
        # STEP 5: GET ONLY THE EVENT ENDPOINTS

        event_preds = predictions[:, self.LabelTypesManager.event_predictions_indicies]

        # STEP 6: CALCULATE THE LOSS FOR EVENT ENDPOINTS
        event_binary_labels_tensor, event_days_label_tensor = self.preprocess_event_target_columns(targets)

        loss_dict_event_endpoints = self.events_loss_function(event_preds, event_days_label_tensor, event_binary_labels_tensor) 

        """
        AGGREGATE THE LOSSES
        """

        # total loss should be the mean over all entries in the loss dicts
        total_loss = torch.stack(list(loss_dict_binary_endpoint.values()) + list(loss_dict_event_endpoints.values())).mean()

        # merge the two dictionaries
        loss_dict = {**loss_dict_binary_endpoint, **loss_dict_event_endpoints}

        return total_loss, loss_dict    




    def preprocess_event_target_columns(self, targets):
        """
        Seperates the event endpoints into a binary 'events' tensor and a 'days' tensor.
        Args:
            targets (torch.Tensor): the target tensor (all of the labels that are used in the model)
            event_label_column_indicies (list): the indicies of the event endpoints in the targets
        Returns:
            binary_events (torch.Tensor): a tensor of binary events (1 for event, 0 for no event). Each column corresponds to a different event endpoint.
            days (torch.Tensor): a tensor of days (the number of days until the event occurs). Each column corresponds to a different event endpoint.
        """
        
        # Get the event endpoints
        event_targets = targets[:, self.LabelTypesManager.event_targets_indicies]

        # Get the binary events and days
        binary_events = event_targets[:, :, 0::2].squeeze(-1)  # Get the binary events (every second column starting from 0)
        days = event_targets[:, :, 1::2].squeeze(-1)  # Get the binary events (every second column starting from 0)

        return binary_events, days



    def forward_binary_loss_function(self, predictions, targets, endpoint_list: list):
        
        predictions = torch.reshape(predictions, targets.shape).to(predictions.dtype)

        # calculate the loss
        batch_loss = self.binary_loss_function(predictions, targets)

        # mask out missing endpoints
        mask = (targets != self.missing_endpoints_as_tensor[0]) # & (targets <= self.valid_endpoints_as_tensor[1])
        batch_loss = torch.nan_to_num(batch_loss, nan=0.0)
        batch_loss *= mask
        
        num_valid_labels = mask.sum() # the number of non-missing labels

        if num_valid_labels == 0: # in case all labels are missing, the loss will be `nan`, so we return 0 instead
            zero_tensor = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)  
            batch_loss_dict = {endpoint:zero_tensor for endpoint in self.config['columns']['labels']}
        
        else:
            # Mean loss per endpoint
            batch_loss_dict = {}

            if len(endpoint_list) > 1:  # If there are multiple endpoints
                for idx, endpoint in enumerate(endpoint_list):
                    n_non_missing = mask[:, idx].sum()  

                    if n_non_missing == 0:  # If there are no non-missing labels, set the loss to 0
                        batch_loss_dict[endpoint] = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

                    else:
                        mean_loss = batch_loss[:, idx].sum() / n_non_missing
                        batch_loss_dict[endpoint] = torch.clamp(mean_loss, min=0, max=10000)  # Optional clamping

            else:  # If there is only one endpoint
                mean_loss = batch_loss[:].sum() / mask[:].sum()
                batch_loss_dict[endpoint_list[0]] = torch.clamp(mean_loss, min=0, max=10000)  # Optional clamping
                

        return batch_loss_dict