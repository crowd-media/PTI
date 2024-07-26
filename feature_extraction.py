import sys
import json
import os

from utils.constants import PTI_FACE_TEMPLATE
from utils.constants import PTI_SIZE

# Dynamically patch the enum module
import enum
from compat import StrEnum
enum.StrEnum = StrEnum

from unith_thai.helpers.detector.dlib_face_detector import DLibFaceDetector
from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader
from unith_thai.helpers.feature.feature_loader import FeatureLoader
from params.feature_extraction_params import FeatureExtractionParams

from video_feature_extractor import VideoFeatureExtractor

def run(params_path: str) -> None:
    print("Starting video features extraction!")
    feature_loader = FeatureLoader()
    with open(params_path) as f:
                print(f"Loading params from {params_path}...")
                params = FeatureExtractionParams(**json.load(f))

    video_reader = StreamVideoReader(params.video_path, loop=False)

    face_detector = DLibFaceDetector(params.landmarks_model_path)
    extractor = VideoFeatureExtractor(
        video_reader,
        face_detector,
        params.mask_params,
        params.key_frames_params,
        params.dbpn.scale_factor,
        params.template_scale_factor,
        smoothing_factor=7,
        face_template = PTI_FACE_TEMPLATE,
        image_size = PTI_SIZE,
    )
    print("Extracting features...")
    features = extractor.extract_features()

    video_name = params.video_path.split("/")[-1]
    feature_loader.save(
        features,
        os.path.join(params.result_path, video_name.split(".")[0]),
    )
    
    return

if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_feature_extraction.json"
    features = run(params_path)

    