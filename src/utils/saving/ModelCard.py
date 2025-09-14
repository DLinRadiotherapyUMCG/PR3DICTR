import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import ProjectDetailsCard

class ModelCard:
    """
    A writable, dictionary-like model card that can be imported
    as a config file anywhere in your project.
    """

    # Default template (acts like a config skeleton)
    _template = {
        "task": "Endpoint",
        "card_metadata": {
            "card_creation_date": str(datetime.now()),
            "version_number": 1.0,
            "version_changes": None,
            "doi": None
        },
        "model_basic_information": {
            "name": "",
            "creation_date": "",
            "version_number": "",
            "version_changes": "",
            "doi": None,
            "model_scope_summary": "",
            "model_scope_anatomical_site": "",
            "clearance_type": None,
            "clearance_approved_by_name": None,
            "clearance_approved_by_institution": None,
            "clearance_approved_by_contact_email": None,
            "clearance_additional_information": None,
            "intended_users": None,
            "observed_limitations": None,
            "potential_limitations": None,
            "type_of_learning_architecture": "",
            "developed_by_name": "",
            "developed_by_institution": "",
            "developed_by_email": "",
            "conflict_of_interest": "",
            "software_license": None,
            "code_source": None,
            "model_source": None,
            "citation_details": None,
            "url_info": None
        },
        "technical_specifications": {
            "model_pipeline_summary": "",
            "model_pipeline_figure_appendix_note": "",
            "model_inputs": [],
            "model_outputs": [],
            "pre-processing": "",
            "post-processing": "",
            "learning_architectures": [
                {
                    "total_number_trainable_parameters": "",
                    "number_of_inputs": "",
                    "input_content_list": [],
                    "output_content_list": [],
                    "loss_function": "",
                    "batch_size": "",
                    "citation_details_ts": "",
                    "id": 0
                }
            ],
            "hw_and_sw": {
                "libraries_and_dependencies": None,
                "hardware_recommended": None,
                "inference_time_for_recommended_hw": None,
                "installation_getting_started": None,
                "environmental_impact": None
            }
        },
        "training_data": {
            "model_name": "NA",
            "url_doi_to_model_card": "NA",
            "tuning_technique": "NA",
            "total_size": "",
            "number_of_patients": "",
            "source": "",
            "acquisition_period": "",
            "inclusion_exclusion_criteria": "",
            "strategy_for_data_augmentation": "",
            "validation_strategy": "",
            "optimiser": ""
        },
        "evaluations": [],
        "other_considerations": {
            "responsible_use_and_ethical_considerations": None,
            "risk_analysis": None,
            "post_market_surveillance_live_monitoring": None
        }
    }

    def __init__(self):
        # Work on a fresh copy of the template
        self.data = deepcopy(self._template)

    # ------------------------
    # Dict-like access
    # ------------------------
    def __getitem__(self, key):
        return self.data[key]

    def setitem(self,key,value):
        self.__setitem__(key,value)

    def __setitem__(self, key, value):
        if(isinstance(key,list)):
            path = key
            d = self.data
            for key in path[:-1]:
                d = d.setdefault(key, {})  # ensures intermediate keys exist
            d[path[-1]] = value
        else:
            self.data[key] = value

    def __repr__(self):
        return f"ModelCard({json.dumps(self.data, indent=2, ensure_ascii=False)})"

    def clean(self):
        self.data = cleandict(self.data)

    def absorb_config_details(self, config):
        # General
        self.__setitem__(["model_basic_information","name"], config["general"]["experiment_name"] + "_" + config["general"]['trialNumber'])
        self.__setitem__(["model_basic_information","version_number"], config["general"]["trialNumber"])
        self.__setitem__(["model_basic_information","type_of_learning_architecture"], config["model"]["model_name"])
        self.__setitem__(["model_basic_information","code_source"], "https://github.com/DLinRadiotherapyUMCG/pred_RT")
        # Technical information
        input = [] + AddPrefix(config["data"]["image_keys"], "3DVolume") + AddPrefix(config["columns"]["clinical_features"],"ClinVAR_")
        self.__setitem__(["technical_specifications","model_inputs"], input)
        output = [] + AddCustomPrefix(config["columns"]["labels"],config["columns"]["labels_types"])
        self.__setitem__(["technical_specifications","model_outputs"], output)
        if(config["data"]["preprocessing"]["isEnabled"]):
            text_preprocessing = []
            if(config["data"]["preprocessing"]["crop"]):
                text_preprocessing.append(f"cropping = {config["data"]['preprocessing']['crop_shape']}")
            scalingInfo = config["data"]['preprocessing']['needs_scaling']
            #text_preprocessing.append(f"needs_scaling = {scalingInfo}")
            if("ct" in scalingInfo):
                text_preprocessing.append(clipRange("ct",config["data"]['preprocessing']['ct']))
            if("rtdose" in scalingInfo):
                text_preprocessing.append(clipRange("rtdose",config["data"]['preprocessing']['rtdose']))
            if("pet" in scalingInfo):
                text_preprocessing.append(clipRange("pet",config["data"]['preprocessing']['pet']))
            if("mri" in scalingInfo):
                text_preprocessing.append(clipRange("mri",config["data"]['preprocessing']['mri']))

            text_preprocessing.append(f"needs_label_mapping = {config["data"]['preprocessing']['needs_label_mapping']}")
            self.__setitem__(["technical_specifications","pre-processing"], text_preprocessing)
        self.__setitem__(["technical_specifications","learning_architectures"],config["model"]["model_name"])   

        aug = config["data"]["augmentation"]
        if(aug["isEnabled"]):
            text_aug = []
            if(aug["list"]["flip"]["isEnabled"]):
                text_aug.append("flip")
            if(aug["list"]["random_crop"]["isEnabled"]):
                text_aug.append("randomCrop")
            if(aug["list"]["affine"]["isEnabled"]):
                text_aug.append(f"Affine: Translate_max = {aug["list"]["affine"]["translate_max"]}, Scale_max = {aug["list"]["affine"]["scale_max"]}, Z_scale = {aug["list"]["affine"]["z_scale"]}")
            if(aug["list"]["rotate"]["isEnabled"]):
                text_aug.append(f"Rotate: Max_angle (degrees) = {aug["list"]["rotate"]["max"]}")
            if(aug["list"]["noise"]["isEnabled"]):
                text_aug.append(f"Noise: mean = {aug["list"]["noise"]["mean"]}, standard deviation = {aug["list"]["noise"]["std"]}")
            if(aug["mixup"]["isEnabled"]):
                text_aug.append(f"MixUp: alpha = {aug["mixup"]["alpha"]}")
        self.__setitem__(["training_data","strategy_for_data_augmentation"],text_aug) 


    def absorb_project_details(self, project: ProjectDetailsCard):
        """Inject shared metadata from a ProjectDetailsCard."""
        pd = project.data
        self.data["model_basic_information"].update({
            "model_scope_summary": pd["model_scope_summary"],
            "intended_users": pd["intended_users"],
            "observed_limitations": pd["observed_limitations"],
            "potential_limitations": pd["potential_limitations"],
            "developed_by_name": pd["developed_by_name"],
            "developed_by_institution": pd["developed_by_institution"],
            "developed_by_email": pd["developed_by_email"],
            "conflict_of_interest": pd["conflict_of_interest"],
        })
        self.data["training_data"].update({
            "source": pd["training_data_source"],
            "acquisition_period": pd["training_data_acquisition_period"],
            "inclusion_exclusion_criteria": pd["training_data_inclusion_exclusion_criteria"],
        })

    # ------------------------
    # Export methods
    # ------------------------
    def to_json(self, filename: str | Path = None) -> str:
        """Return JSON as a string or write to file if filename is given."""
        json_str = json.dumps(self.data, indent=2, ensure_ascii=False)
        if filename:
            Path(filename).write_text(json_str, encoding="utf-8")
        return json_str

    def update(self, path: list[str], value):
        """
        Update a nested key using a list as a path.
        Example:
            card.update(["model_basic_information", "doi"], "10.1000/example")
        """
        d = self.data
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = value


def AddPrefix(listItems, prefix):
    newList = []
    for i in range(len(listItems)):
        newList.append(prefix + "_" + str(listItems[i]))
    return newList

def AddCustomPrefix(listItems, prefixList):
    newList = []
    for i in range(len(listItems)):
        newList.append(prefixList[i] + "_" + str(listItems[i]))
    return newList

def clipRange(typeGiven, config_ClippingInfo):
    return f"{typeGiven} clipped values [{config_ClippingInfo['a_min']} - {config_ClippingInfo['a_max']} to format [{config_ClippingInfo['b_min']} - {config_ClippingInfo['b_max']}]"

def cleandict(data):
    keys = data.keys()
    for key in keys:
        if(isinstance(data[key], dict)):
            data[key] = cleandict(data[key])
        else:
            data[key] = None
    return data