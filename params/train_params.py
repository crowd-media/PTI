from typing import Optional
from pydantic import BaseModel

from unith_thai.cli.params.video_params import VideoParams

class TrainParams(BaseModel):
    video_params: VideoParams
    features_path: str
    duration: Optional[int] = 10
    intensity: Optional[float] = 2.0
    use_multi_id_training: Optional[bool] = False
    use_last_w_pivots: Optional[bool] = False
