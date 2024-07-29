from typing import Optional
from pydantic import BaseModel

class TrainParams(BaseModel):
    video_path: str
    features_path: str
    duration: int
    use_multi_id_training: bool 
    use_last_w_pivots: bool
