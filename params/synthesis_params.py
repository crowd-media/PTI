from typing import Optional
from pydantic import BaseModel

class SynthesisParams(BaseModel):
    video_path: str
    result_path: str
    duration: int
    intensity: float
    model_id: str
    use_multi_id_training: Optional[bool] = False
