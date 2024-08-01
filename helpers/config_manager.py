import json
import os

from pydantic_core import PydanticUndefinedType

from params.feature_extraction_params import FeatureExtractionParams
from params.train_params import TrainParams
from params.synthesis_params import SynthesisParams
from params.blend_params import BlendParams

class ConfigManager():
    def __init__(self, config_path):
        self.base_dir = os.path.dirname(config_path)

        self.steps = ['train', 'synthesize', 'blend']
        self.params = [TrainParams, SynthesisParams, BlendParams]

        self.create_configs()
            

    def create_configs(self):
        # Create config files for each step

        for step, param in zip(self.steps, self.params):
            config_data = self.create_config_data(param)
            self.save_config(step, config_data)
        return

    def create_config_data(self, param):
        #from params create config file
        config_data = {}
        for field in param.__fields__.keys():
            config_data[field] = param.__fields__[field].default 
            if isinstance(config_data[field], PydanticUndefinedType):
                config_data[field] = None
        return config_data
    
    def save_config(self, step, config_data):
        with open(os.path.join(self.base_dir, f"config_{step}.json"), 'w') as f:
            json.dump(config_data, f, indent=4)

    def open_config(self, step):
        config_name = f"config_{step}.json"
        config_path = os.path.join(self.base_dir, config_name)
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data


    def update_config(self, config_data, key, value):        
        config_data[key] = value
        return config_data