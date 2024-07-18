import json
import os.path

import numpy as np

from unith_thai.helpers.feature.io.talking_head_features import TalkingHeadFeatures

class FeatureLoader:
    @staticmethod
    def save(features: TalkingHeadFeatures, save_path: str) -> None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(
            os.path.join(save_path, "affine_matrix.npy"),
            features.affine_matrix,
        )
        np.save(
            os.path.join(save_path, "inverse_affine_matrix.npy"),
            features.inverse_affine_matrix,
        )
        np.save(
            os.path.join(save_path, "mask.npy"),
            features.mask,
        )
        np.save(
            os.path.join(save_path, "crop_coordinates.npy"),
            features.crop_coordinates,
        )
        if features.key_frames and len(features.key_frames) > 0:
            with open(
                os.path.join(save_path, "key_frames.json"), "w"
            ) as f:
                json.dump(features.key_frames, f, indent=4)

    @staticmethod
    def load(features_path: str) -> TalkingHeadFeatures:
        if not os.path.exists(features_path):
            raise FileNotFoundError("Wrong path, cannot load params")
        return TalkingHeadFeatures(
            affine_matrix=np.load(
                os.path.join(features_path, "affine_matrix.npy")
            ),
            inverse_affine_matrix=np.load(
                os.path.join(
                    features_path, "inverse_affine_matrix.npy"
                )
            ),
            mask=np.load(os.path.join(features_path, "mask.npy")),
            crop_coordinates=np.load(
                os.path.join(features_path, "crop_coordinates.npy")
            ),
            key_frames={},
        )