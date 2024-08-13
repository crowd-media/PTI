import json
import os
from typing import Optional
from unith_thai.helpers.detector.dlib_face_detector import DLibFaceDetector
from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader
from unith_thai.helpers.feature.feature_loader import FeatureLoader

# Dynamically patch the enum module (python 3.10 compatibility)
import enum
from compat import StrEnum
enum.StrEnum = StrEnum

from helpers.config_manager import ConfigManager
from utils.constants import PTI_FACE_TEMPLATE
from utils.constants import PTI_SIZE
from params.feature_extraction_params import FeatureExtractionParams
from video_feature_extractor import VideoFeatureExtractor

def extract_features(params_path: str, config_manager = Optional[ConfigManager]) -> None:
    print("Starting video features extraction!")
    feature_loader = FeatureLoader()
    with open(params_path) as f:
                print(f"Loading params from {params_path}...")
                params = FeatureExtractionParams(**json.load(f))

    video_reader = StreamVideoReader(params.video_params.video_path, loop=False)

    face_detector = DLibFaceDetector(params.landmarks_model_path)
    extractor = VideoFeatureExtractor(
        video_reader,
        face_detector,
        params.mask_params, 
        scale_factor = params.scale_factor,
        face_template = PTI_FACE_TEMPLATE,
        image_size = PTI_SIZE,
    )
    print("Extracting features...")
    features = extractor.extract_features()

    feature_loader.save(
        features,
        params.features_path,
    )
    print(f"Features saved in {params.features_path}")

if __name__ == "__main__":
    params_path = "/home/ubuntu/efs/data/users/itziar/config_files/PTI/config_feature_extraction.json"
    features = extract_features(params_path)

    