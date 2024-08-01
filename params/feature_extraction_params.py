from pydantic import BaseModel
from pydantic import Field

from unith_thai.cli.params.mask_params import MaskParams

class FeatureExtractionParams(BaseModel):
    video_path: str
    landmarks_model_path: str
    result_path: str

    scale_factor: int = Field(default=1)
    mask_params: MaskParams
    template_scale_factor: float = Field(default=1)
    features_path: str = Field(default="")
