from typing import Optional
from pydantic import BaseModel

from unith_thai.cli.params.video_params import VideoParams

class SynthesisParams(BaseModel):
    video_params: VideoParams
    synth_result_path: Optional[str] = "./results"
    duration: Optional[int] = 10
    intensity: Optional[float] = 2.0
    model_id: str
    use_multi_id_training: Optional[bool] = False
