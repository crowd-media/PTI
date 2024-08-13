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
    
    
if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_synthesize.json"
    synthesize(params_path, None)