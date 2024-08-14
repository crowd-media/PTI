from pydantic import BaseModel
from pydantic import Field

from unith_thai.cli.params.mask_params import MaskParams
from unith_thai.cli.params.video_params import VideoParams

class FeatureExtractionParams(BaseModel):
    video_params: VideoParams
    landmarks_model_path: str
    features_path: str
    scale_factor: int = Field(default=1)
    mask_params: MaskParams
