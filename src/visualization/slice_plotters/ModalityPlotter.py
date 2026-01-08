



class ModalityPlotter:
    """ Base class for modality plotting strategies """
    def __init__(self, config, modality_name):

        self.modality_config = config['data']['preprocessing'][modality_name]
        self.modality_name = modality_name
        self.RT_region = config['general']['region']
    
        self.legend_title = self.modality_config.get('legend_title', None)
        self.set_colourmaps()

        

    def set_colourmaps(self):
        self.cmap = None
        self.norm = None
        self.levels = None
        

    def plot(self, axs, data, slices, params, **kwargs):
        raise NotImplementedError