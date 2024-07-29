import json
from params.train_params import TrainParams
from trainer import Trainer
import time

from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader

from helpers.feature_loader import FeatureLoader

def run(params_path: str) -> None:
    start_time = time.time()
    print("Starting synthesis command!")
    with open(params_path) as f:
                print(f"Loading params from {params_path}...")
                params = TrainParams(**json.load(f))

    video_reader = StreamVideoReader(params.video_path, loop=False)

    print(f"Loading features from {params.features_path}...")
    feature_loader = FeatureLoader()
    features = feature_loader.load(params.features_path)

    synthesizer = Trainer(
        params.video_path,
        features,
        params.duration,
        params.use_last_w_pivots,
        video_reader,
        params.use_multi_id_training
        )
    
    print(f"Synthesizing {params.video_path}...")
    synthesizer.synthesize()
    end_time = time.time()  # Record the end time
    print(f"Execution time: {end_time - start_time} seconds")  # Print the execution time
    
    return

if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_train.json"
    run(params_path)