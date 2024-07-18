from typing import Optional
from pydantic import BaseModel

class SynthesisParams(BaseModel):
    video_path: str
    features_path: str
    result_path: str
    duration: int
    intensity: float
    use_multi_id_training: bool 
    use_last_w_pivots: bool
