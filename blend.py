import json

# Dynamically patch the enum module (python 3.10 compatibility)
import enum
from compat import StrEnum
enum.StrEnum = StrEnum

from unith_thai.helpers.feature.feature_loader import FeatureLoader
from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader

from params.blend_params import BlendParams
from blend_pipeline import BlendPipeline
from dataloaders.dataloader import BlendDataLoader
from dataloaders.imageloader import ImageLoader
from helpers.ffmpeg_video_writer import FfmpegVideoWriter

def blend(params_path: str) -> None:
    print("Starting blending")
    with open(params_path) as f:
        print(f"Loading params from {params_path}...")
        params = BlendParams(**json.load(f))

    print(f"Loading features from {params.features_path}...")
    feature_loader = FeatureLoader()
    features = feature_loader.load(params.features_path)

    face_loader = ImageLoader()

    print(f"Loading generated faces from {params.synth_result_path}...")
    faces = face_loader.load(params.synth_result_path)

    print("Preparing Ffmpeg Video Writer...")
    frame_writer = FfmpegVideoWriter(
        params.result_path,
        params.video_params.video_size_height,
        params.video_params.video_size_width,
        params.video_params.fps,
        start_frame=params.video_params.start_frame,
    )

    pipeline = BlendPipeline(frame_writer)

    video_reader = StreamVideoReader(params.video_params.video_path, False)

    print("Preparing data loader...")
    data_loader = BlendDataLoader(
        video_reader,
        features,
        faces,
        params.blend_batch_size,
    )
    print(f"Data Loader ready with {len(data_loader)} elements!")

    print("Start processing data...")
    pipeline.process(data_loader)
    print("Blending data processed!")
    return

if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_blend.json"
    blend(params_path)