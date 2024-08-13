import json
from typing import Optional
import time
from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader


from helpers.feature_loader import FeatureLoader
from helpers.config_manager import ConfigManager
from params.train_params import TrainParams
from trainer import Trainer


def train(params_path: str, config_manager: Optional[ConfigManager]) -> None:
    start_time = time.time()
    print("Starting training")

    with open(params_path) as f:
                print(f"Loading params from {params_path}...")
                params = TrainParams(**json.load(f))

    video_reader = StreamVideoReader(params.video_params.video_path, loop=False)

    print(f"Loading features from {params.features_path}...")
    feature_loader = FeatureLoader()
    features = feature_loader.load(params.features_path)

    trainer= Trainer(
        params.video_params.video_path,
        features,
        params.duration,
        params.use_last_w_pivots,
        video_reader,
        params.use_multi_id_training
        )
    
    print(f"Training {params.video_params.video_path}...")
    model_id = trainer.train()
    end_time = time.time()  # Record the end time
    print(f"Training execution time: {end_time - start_time} seconds")  # Print the execution time

    if config_manager is None:
        return model_id
    else:
        config_data = config_manager.open_config("synthesize")
        config_data = config_manager.update_config(config_data, "model_id", model_id)     
        config_manager.save_config("synthesize", config_data)
        return model_id
    

if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_train.json"
    
    train(params_path)