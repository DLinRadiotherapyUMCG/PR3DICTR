# Create a colormap for the negative side of the attention map
neg_cmap = fade_colormap(cc.CET_R4, 0)
neg_cmap = list(reversed(neg_cmap))

# Create a colormap for the positive side of the attention map
pos_cmap = fade_colormap(cc.CET_R4, 0)

# Append positive and negative colormaps to create a single colormap
att_cmap = np.concatenate((neg_cmap, pos_cmap))
# Keep only the positive side for abosulte attention maps
att_cmap_abs = pos_cmap




HNC_plotting_params = {
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
}


LUNG_plotting_params = {
    "CT": {
        "cmap": "gray",
        "cmap_title": "HU",
        "min_val": -1200,
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
}