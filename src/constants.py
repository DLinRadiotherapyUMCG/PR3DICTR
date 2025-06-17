import torch

from src.visualization.attention_colormap import att_cmap, att_cmap_abs

MAIN_DATA_SOURCE = "UMCG"

# Dataset column names
SPLIT_COL_NAME = "Split"
PATIENT_ID_COL_NAME = "PatientID"

PATIENT_ID_LENGTHS_DICT = {
    "UMCG": 7,
    "MDACC" : 10,
}
MISSING_DATA_VALUE = -1   # value for missing endpoints

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VERBOSE = True


METRIC_TYPES = {
    "Classification" : [
        'AUC',
        'AUC_se',
        'accuracy',
        'balanced_accuracy',
        'F1_score',
        'precision',
        'recall',
    ],
    "Regression" : [
        'MSE',
        'RMSE',
        'MAE',
        'r2',
    ],
    "Calibration" : [
        'brier_score',
        'ECE',
        'MCE',
        'ACE',
    ],
}




""" The dictionary of colour mapping for HNC RTDOSE  """
# keys: dose value thresholds
# values: RGB colour tuples
PLOT_SLICES_COLOURMAPS= {
    "HNC" : {
        8400: (64,0,0),
        8050: (128,0,0),
        7700: (128,0,0),
        7490: (255,0,0),
        7000: (255,128,0),
        6650: (255,128,0),
        6300: (255,255,0),
        5805: (0,64,0),
        5425: (0,255,0),
        5154: (0,255,0),
        5000: (0,255,255),
        4883: (0,255,255),
        4000: (0,128,192),
        3000: (0,0,160),
        2000: (0,0,160),
        1000: (192,192,192),
        500: (192,192,192),
        200: (255,255,255),
        0: (255,255,255),
    },
    "LUNG" : {
        8000: (248,0,33),# (248,0,33),  # NOTE: Daniel added this extra upper limit. Ensure that it's correct
        6900: (248,0,33),
        6700: (248,0,33),
        6420: (255,118,74),
        6000: (46,255,13),
        5700: (46,255,13),
        5400: (66,255,253),
        5000: (251,255,35),
        4000: (27,62,126),
        3000: (5,0,253),
        2000: (5,0,253),
        1000: (245,245,245),
        500: (255,255,255),
        200: (255,255,255),
    }
}






PLOTTING_PARAMS = {
    "HNC" : {
        "CT": {
            "cmap": "gray",
            "cmap_title": "HU",
            "min_val": -200,
            "max_val": 400,
        },
        "RTDOSE": {
            "cmap": "dose",
            "cmap_title": "Dose (Gy)",
            "min_val": 0,
            "max_val": 8000,
        },
        "RTSTRUCT": {"color": "deeppink", "linewidth": 2, "alpha": 0.8, "cmap": "gray"},
        "Attention": {
            "cmap": "Attention",
            "cmap_abs": "AttentionAbs",
            "cmap_colors": att_cmap,
            "cmap_abs_colors": att_cmap_abs,
            "cmap_title": None,
            "alpha": 1,
            "background_color": "black",
        },
    },

    "LUNG" : {
        "CT": {
            "cmap": "gray",
            "cmap_title": "Binary",
            "min_val": 0,
            "max_val": 1,
        },
        "RTDOSE": {
            "cmap": "dose",
            "cmap_title": "Dose (Gy)",
            "min_val": 0,
            "max_val": 8000,
        },
        "RTSTRUCT": {"color": "deeppink", "linewidth": 2, "alpha": 0.8, "cmap": "gray"},
        "Attention": {
            "cmap": "Attention",
            "cmap_abs": "AttentionAbs",
            "cmap_colors": att_cmap,
            "cmap_abs_colors": att_cmap_abs,
            "cmap_title": None,
            "alpha": 1,
            "background_color": "black",
        },
    }

}
