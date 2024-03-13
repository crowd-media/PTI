import json
import sys

sys.path.append('/home/ubuntu/talking-heads-ai')
from unith_thai.helpers.feature.feature_loader import FeatureLoader
from unith_thai.data_loaders.frame_reader import FrameReader
from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader


from params.blend_params import BlendParams
from blend_pipeline import BlendPipeline
from dataloaders.dataloader import BlendDataLoader
from dataloaders.imageloader import ImageLoader
from ffmpeg_video_writer import FfmpegVideoWriter

def run(params_path: str) -> None:
    print("Starting Talking Head inference command!")
    with open(params_path) as f:
        print(f"Loading params from {params_path}...")
        params = BlendParams(**json.load(f))

    print(f"Loading features from {params.features_path}...")
    feature_loader = FeatureLoader()
    features = feature_loader.load(params.features_path)

    face_loader = ImageLoader()

    print(f"Loading generated faces from {params.generated_faces_path}...")
    faces = face_loader.load(params.generated_faces_path)

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
        params.batch_size,
    )
    print(f"Data Loader ready with {len(data_loader)} elements!")

    print("Start processing data...")
    pipeline.process(data_loader)
    
    return

if __name__ == "__main__":
    params_path = "/home/ubuntu/PTI/config_blend.json"
    run(params_path)