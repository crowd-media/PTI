from pydantic import BaseModel
from pydantic import Field
import sys


from unith_thai.cli.params.dbpn_params import DBPNParams
from unith_thai.cli.params.mask_params import MaskParams
from unith_thai.cli.params.key_frames_params import KeyFramesParams


class FeatureExtractionParams(BaseModel):
    video_path: str
    landmarks_model_path: str
    result_path: str

    dbpn: DBPNParams
    mask_params: MaskParams
    key_frames_params: KeyFramesParams
    template_scale_factor: float = Field(default=1)
    features_path: str = Field(default="")
