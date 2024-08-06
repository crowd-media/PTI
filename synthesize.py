import json
from typing import Optional
import time
from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader

from params.synthesis_params import SynthesisParams
from synthesizer import Synthesizer
from helpers.config_manager import ConfigManager


def synthesize(params_path: str, config_manager: Optional[ConfigManager]) -> None:
    start_time = time.time()
    print("Starting synthesis")
    with open(params_path) as f:
                print(f"Loading params from {params_path}...")
                params = SynthesisParams(**json.load(f))

    synthesizer = Synthesizer(
        params.video_params.video_path, 
        params.synth_result_path,
        params.duration,
        params.intensity,
        params.model_id,
        )
    
    print(f"Synthesizing {params.video_params.video_path}...")
    synthesizer.synthesize()
    end_time = time.time()  # Record the end time
    print(f"Synthesis execution time: {end_time - start_time} seconds")  # Print the execution time
    
    if config_manager is None:
        return
    else:
        config_data = config_manager.open_config("blend")
        data_id = params.video_params.video_path.split("/")[-1].split(".")[0]
        result_path = f'{params.synth_result_path}/{data_id}/{params.duration}_{params.intensity}'
        config_data = config_manager.update_config(config_data, "synth_result_path", result_path)     
        config_manager.save_config("blend", config_data)
        return
    
if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_synthesize.json"
    synthesize(params_path, None)