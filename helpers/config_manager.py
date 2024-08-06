import json
import os
from typing import Any, Dict, Type

from pydantic import BaseModel
from pydantic_core import PydanticUndefinedType

from params.feature_extraction_params import FeatureExtractionParams
from params.train_params import TrainParams
from params.synthesis_params import SynthesisParams
from params.blend_params import BlendParams

class ConfigManager():
    def __init__(self, base_path: str) -> None:
        base_config_path = os.path.join(base_path, "config.json")

        with open(base_config_path, 'r') as f:
            base_config_data = json.load(f)
        
        self.base_path = base_path

        self.steps = ['feature_extraction', 'train', 'synthesize', 'blend']
        self.params = [FeatureExtractionParams, TrainParams, SynthesisParams, BlendParams]

        self.create_configs(base_config_data)

    def create_configs(self, base_config_data: str) -> None:
        # Create config files for each step
        for step, param in zip(self.steps, self.params):
            config_data = self.create_config_data(param, base_config_data)
            self.save_config(step, config_data)
        return

    def create_config_data(self,  param: Type[BaseModel], base_config_data) -> Dict[str, Any]:
        #from params create config file
        config_data = {}
        for field in param.model_fields.keys():
            config_data[field] = param.model_fields[field].default 
            if isinstance(config_data[field], PydanticUndefinedType):
                config_data[field] = None

        #fill fields with base_config data
        for key, value in base_config_data.items():
            if key in config_data:
                config_data[key] = value
        return config_data
    
    def save_config(self, step, config_data: Dict) -> None:
        with open(os.path.join(self.base_path, f"config_{step}.json"), 'w') as f:
            json.dump(config_data, f, indent=4)

    def open_config(self, step: str) -> Dict:
        config_name = f"config_{step}.json"
        config_path = os.path.join(self.base_path, config_name)
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return config_data


    def update_config(self, config_data: Dict, key: str, value: Any) -> Dict:        
        config_data[key] = value
        return config_data