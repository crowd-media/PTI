from pydantic import BaseModel
import sys

from unith_thai.cli.params.mask_params import MaskParams
from unith_thai.cli.params.video_params import VideoParams


class BlendParams(BaseModel):
    features_path: str
    synth_result_path: str
    result_path: str
    mask_params: MaskParams
    video_params: VideoParams
    blend_batch_size: int
