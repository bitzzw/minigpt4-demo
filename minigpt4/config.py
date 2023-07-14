from omegaconf import OmegaConf


class Config:
    def __init__(self, cfg_path):

        self.config = OmegaConf.load(cfg_path)

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model
    
    @property
    def preprocess_cfg(self):
        return self.config.preprocess
