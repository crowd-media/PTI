from pydantic import BaseModel
import sys

sys.path.append('/home/ubuntu/talking-heads-ai')
from unith_thai.cli.params.mask_params import MaskParams
from unith_thai.cli.params.video_params import VideoParams
from unith_thai.cli.params.audio_params import AudioParams


class BlendParams(BaseModel):
    features_path: str
    generated_faces_path: str
    result_path: str
    mask_params: MaskParams
    video_params: VideoParams
    batch_size: int
