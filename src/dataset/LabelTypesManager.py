from collections import defaultdict



class LabelTypesManager(object):
    """
    A class to handle the label types in the dataset.
    It checks if the label types are valid and returns the indices of the binary and event labels.
    """
    def __init__(self, config):
        self.config = config

        self.labels_list = config['columns']['labels']
        self.label_types = config['columns']['labels_types']
        

        self.label_names_full_list, self.label_types_full_list = self.check_label_types(self.labels_list, self.label_types)
        

        self.endpoint_type_groups_indicies, self.endpoint_type_groups_names = self.seperate_endpoint_indicies(self.labels_list, self.label_types)

        #print(self.endpoint_type_groups_indicies)

        # set the binary endpoint indicies for the labels tensors and the predictions tensors
        self.binary_targets_indicies = self.endpoint_type_groups_indicies['Binary']
        self.binary_predictions_indicies = [i for i, t in enumerate(self.config['columns']['labels_types']) if t == "Binary"]

        # set the event endpoint indicies for the labels tensors and the predictions tensors
        self.event_targets_indicies = self.endpoint_type_groups_indicies['Event']
        self.event_predictions_indicies = [i for i, t in enumerate(self.config['columns']['labels_types']) if t == "Event"]

    
    def check_label_types(self, endpoint_list: list, label_types: list):
        """
        Check if the label types in the config match the endpoint list.
        Args:
            config (dict): configuration object
            endpoint_list (list): list of endpoints to check
        Returns:
            output_endpoint_list (list): a list of endpoints with the correct label types
            output_label_types (list): a list of label types corresponding to the endpoints in the output_endpoint_list
        Example:
            endpoint_list = ['Xerostomia', 'Dysphagia', 'OS', 'DMFS']
            label_types = ['Binary', 'Binary', 'Event', 'Event']
            output_endpoint_list = ['Xerostomia', 'Dysphagia', 'OS_event', 'OS_days', 'DMFS_event', 'DMFS_days']
            output_label_types = ['Binary', 'Binary', ('Event', 'Days'), ('Event', 'Days')]

        """

        output_endpoint_list = []
        output_label_types = []
        
        for endpoint_name, label_type in zip(endpoint_list, label_types):
            if label_type == "Binary":
                output_endpoint_list.append(endpoint_name)
                output_label_types.append("Binary")
            elif label_type == "Event":
                # for event endpoints, we need to add the '_days' suffix and '_event' to the endpoint name
                #output_endpoint_list.append(endpoint_name + "_event")
                output_endpoint_list.append((endpoint_name + "_event", endpoint_name + "_days"))
                output_label_types.append( ("Event", "Days") )
                #output_label_types.append("Event")
                #output_endpoint_list.append(endpoint_name + "_days")
            else:
                raise ValueError(f"Invalid label type: {label_type}. Must be 'Binary' or 'Event'.")
        

        return output_endpoint_list, output_label_types


    def seperate_endpoint_indicies(self, endpoint_list, endpoint_types_list):
        """
        A function to identify which indicies within the labels tensor correspond to which endpoint type ('Binary' or 'Event').
        Args:
            endpoint_list (list): a list of endpoints (labels) in the dataset
            endpoint_types_list (list): a list of endpoint types (e.g. 'Binary', 'Event') corresponding to the endpoints in the endpoint_list
        Returns:
            endpoint_type_groups_indicies (dict): a dictionary where the keys are the endpoint types and the values are lists of indicies corresponding to that endpoint type.
            endpoint_type_groups_names (dict): a dictionary where the keys are the endpoint types and the values are lists of endpoint names corresponding to that endpoint type.
        Example:
            endpoint_list = ['Xerostomia', 'Dysphagia', 'OS', 'DMFS']
            endpoint_types_list = ['Binary', 'Binary', 'Event', 'Event']
            endpoint_type_groups_indicies = {
                'Binary': [0, 1],
                'Event': [[2, 3], [4, 5]]
            }
            endpoint_type_groups_names = {
                'Binary': ['Xerostomia', 'Dysphagia'],
                'Event': ['OS', 'DMFS']
            }
        """

        endpoint_type_groups_indicies = defaultdict(list)
        endpoint_type_groups_names = defaultdict(list)


        idx = 0
        for endpoint, endpoint_type in zip(endpoint_list, endpoint_types_list):
        #print(endpoint_type)
            if 'Event' == endpoint_type:
                endpoint_type_groups_indicies[endpoint_type].append([idx, idx + 1])     # NOTE: add two indicies; one for event and one for days
            #endpoint_type_groups_indicies[endpoint_type].append(idx + 1)
            
                endpoint_type_groups_names[endpoint_type].append(endpoint)
            #endpoint_type_groups_names[endpoint_type].append(endpoint_list[idx + 1])

                idx += 2
            else:
                endpoint_type_groups_indicies[endpoint_type].append(idx)
                endpoint_type_groups_names[endpoint_type].append(endpoint)
                idx += 1

        endpoint_type_groups_indicies = dict(endpoint_type_groups_indicies)
        endpoint_type_groups_names = dict(endpoint_type_groups_names)

        return endpoint_type_groups_indicies, endpoint_type_groups_names