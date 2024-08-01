from typing import Optional
from pydantic import BaseModel

class SynthesisParams(BaseModel):
    video_path: str
    result_path: str
    duration: Optional[int] = 10
    intensity: Optional[float] = 2.0
    model_id: str
    use_multi_id_training: Optional[bool] = False
