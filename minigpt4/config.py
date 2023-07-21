from omegaconf import OmegaConf


class Config:
    def __init__(self, cfg_path):

        self.config = OmegaConf.load(cfg_path)

    @property
    def vis_processor_cfg(self):
        return self.config.vis_processor

    @property
    def model_cfg(self):
        return self.config.model
    
