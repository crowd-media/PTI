import logging
from typing import List
from typing import Tuple

import cv2
import numpy as np
import sys

sys.path.append('/home/ubuntu/talking-heads-ai')
from unith_thai.cli.params.key_frames_params import KeyFramesParams
from unith_thai.cli.params.mask_params import MaskParams
from unith_thai.data_loaders.frame_reader import FrameReader
from unith_thai.helpers.detector.face_detector import FaceDetector
from unith_thai.helpers.feature.image_feature_extractor import (
    ImageFeatureExtractor,
)
from unith_thai.helpers.feature.io.face_coordinates import FaceCoordinates
from unith_thai.helpers.feature.io.face_landmarks import FaceLandmarks
from unith_thai.helpers.feature.io.talking_head_features import TalkingHeadFeatures
from unith_thai.metrics.metrics import Metric
from unith_thai.metrics.ssim_metric import SSIMMetric
from unith_thai.utils.accumulated_time import execution_time_decorator
from unith_thai.utils.constants import FPS
from unith_thai.utils.constants import W2L_FACE_TEMPLATE
from unith_thai.utils.constants import W2L_SIZE
from unith_thai.utils.facial_utils import scale_face_template

from optimal_crop_calculation import extend_crop_margins

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    def __init__(
        self,
        video_reader: FrameReader,
        face_detector: FaceDetector,
        mask_params: MaskParams,
        key_frames_params: KeyFramesParams,
        scale_factor: int = 4,
        template_scale_factor: float = 1,
        smoothing_factor: int = 7,
        face_template: FaceLandmarks = W2L_FACE_TEMPLATE,
        image_size: Tuple = W2L_SIZE,
    ):
        self.video_reader = video_reader
        self.w2l_face_template_scaled = scale_face_template(
            face_template, image_size, template_scale_factor
        )
        self.image_feature_extractor = ImageFeatureExtractor(
            face_detector,
            self.w2l_face_template_scaled,
            scale_factor,
            mask_params.mask_offset_factor,
            mask_params.mask_sigma,
            mask_params.mask_diameter_scale,
        )
        self.key_frames_params = key_frames_params
        self.smoothing_factor = smoothing_factor
        self.face_size = (1024,1024)
        self.scale_factor = 1
        self.sr_area_coordinates = FaceCoordinates(
            left=0,
            right=1024,
            top=0,
            bottom=1024,
        )
        self.mask_params = mask_params

    def get_smooth_affine_inverse_and_mask(
        self, affine_matrices: np.ndarray, crop_coordinates: FaceCoordinates
    ) -> TalkingHeadFeatures:
        """
        Returns averaged bounding boxes
        inputs:
            boxes: list
                array of bounding boxes to smooth
            T: int
                amount of frames to consider on smoothing
        outputs:
            boxes: list
                modified array with smooth bounding boxes
        """

        inverse_affine_matrices = np.zeros(affine_matrices.shape)
        masks = np.zeros(
            (
                affine_matrices.shape[0],
                crop_coordinates.height(),
                crop_coordinates.width(),
            ),
            dtype=np.uint8,
        )

        mouth_mask_template = self.compute_mouth_mask_template()
        for i in range(len(affine_matrices)):
            if i + self.smoothing_factor > len(affine_matrices):
                window = affine_matrices[len(affine_matrices) - self.smoothing_factor :]
            else:
                window = affine_matrices[i : i + self.smoothing_factor]
            affine_matrices[i] = np.mean(window, axis=0)

            inverse_affine_matrices[i] = cv2.invertAffineTransform(affine_matrices[i])
            masks[i] = np.multiply(
                cv2.warpAffine(
                    mouth_mask_template,
                    inverse_affine_matrices[i],
                    (crop_coordinates.width(), crop_coordinates.height()),
                ),
                255,
            )
        talking_head_features = TalkingHeadFeatures(
            affine_matrix=affine_matrices,
            inverse_affine_matrix=inverse_affine_matrices,
            mask=masks,
            crop_coordinates=crop_coordinates.to_array(),
            key_frames={},
        )

        return talking_head_features
    
    def compute_mouth_mask_template(
        self,
    ) -> np.ndarray:
        mask = np.zeros(
            (
                self.face_size[0] * self.scale_factor,
                self.face_size[1] * self.scale_factor,
            ),
            dtype=np.float32,
        )
        top = self.sr_area_coordinates.top * self.scale_factor
        bottom = self.sr_area_coordinates.bottom * self.scale_factor
        left = self.sr_area_coordinates.left * self.scale_factor
        right = self.sr_area_coordinates.right * self.scale_factor

        # Calculate offset based on the image dimensions
        offset_x = int((right - left) * self.mask_params.mask_offset_factor)
        offset_y = int((bottom - top) * self.mask_params.mask_offset_factor)

        # Compute grid_x and grid_y based on the image dimensions and offset
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1, 1, right - left - 2 * offset_x),
            np.linspace(-1, 1, bottom - top - 2 * offset_y),
        )

        diameter = np.sqrt(grid_x**2 + grid_y**2) * self.mask_params.mask_diameter_scale

        kernel = np.exp(-((diameter**2) / (2.0 * self.mask_params.mask_sigma**2)))

        mask[
            top + offset_y : bottom - offset_y, left + offset_x : right - offset_x
        ] = kernel
        return mask

    @execution_time_decorator(__name__)
    def extract_features(self) -> TalkingHeadFeatures:
        face_coordinates_list = self.get_frame_face_coordinates()
        optimal_crop = self.calculate_optimal_crop(face_coordinates_list)
        logger.info(f"Optimal crop coordinates: {optimal_crop}")
        factor = 1.5
        optimal_crop = extend_crop_margins(optimal_crop, factor)
        logger.info(f"Optimal crop coordinates: {optimal_crop}")
        face_landmarks_list = self.get_smooth_landmarks(optimal_crop, window_size=self.smoothing_factor)
        affine_matrices = []
        for face_landmarks in face_landmarks_list:
            affine_matrix = self.image_feature_extractor.get_affine_matrix(
                face_landmarks
            )
            affine_matrices = (
                affine_matrix
                if len(affine_matrices) == 0
                else np.concatenate((affine_matrices, affine_matrix), axis=0)
            )

        features: TalkingHeadFeatures = self.get_smooth_affine_inverse_and_mask(
            affine_matrices, optimal_crop
        )
        if self.key_frames_params.activate_key_frames:
            # set keyframes
            features.key_frames = self.get_key_frames(optimal_crop)
        return features

    @staticmethod
    def calculate_optimal_crop(coordinates: List[FaceCoordinates]) -> FaceCoordinates:
        logger.info("Computing optimal crop...")
        min_left = min(coordinates, key=lambda face: face.left).left
        max_right = max(coordinates, key=lambda face: face.right).right
        min_top = min(coordinates, key=lambda face: face.top).top
        max_bottom = max(coordinates, key=lambda face: face.bottom).bottom
        return FaceCoordinates(
            left=min_left, right=max_right, top=min_top, bottom=max_bottom
        )

    @execution_time_decorator(__name__)
    def get_frame_face_coordinates(self) -> List[FaceCoordinates]:
        logger.info("Obtaining frame coordinates")
        coordinates_list = []
        for image in self.video_reader:
            coordinates_list.append(
                self.image_feature_extractor.get_crop_coordinates(image)
            )
        return coordinates_list

    @execution_time_decorator(__name__)
    def get_key_frames(self, optimal_crop: FaceCoordinates) -> dict[float, float]:
        logger.info("Obtaining key frames")
        # TODO: Generalize the metric in order to change it in the future
        ssim_metric = SSIMMetric()
        crops_list = self.get_crop_frames(optimal_crop)
        similarity_matrix = self.get_similarity_matrix(crops_list, ssim_metric)
        key_frames = self.key_frames_constraints(similarity_matrix)
        return key_frames

    @execution_time_decorator(__name__)
    def get_similarity_matrix(
        self,
        crops_list: List[np.ndarray],
        metric: Metric,
    ) -> np.ndarray:
        logger.info("Obtaining similarity matrix")

        similarity_matrix = np.zeros(
            (self.video_reader.total_frames, self.video_reader.total_frames),
            dtype=np.float16,
        )
        for i in range(0, len(crops_list), self.key_frames_params.sampling_frames):
            for j in range(i, len(crops_list), self.key_frames_params.sampling_frames):
                similarity_matrix[i, j] = self.get_metric_value(
                    metric, crops_list[i], crops_list[j]
                )
        return similarity_matrix


    def key_frames_constraints(
        self, similarity_matrix: np.ndarray
    ) -> dict[float, float]:
        """ " Get the similarity matrix and apply the the filters to get the jump frame for each frame"""
        key_frames = {}
        for frame_idx, frame_ssim in enumerate(similarity_matrix):
            # Find the indices where the values are greater than the threshold
            selected_frames = np.where(
                frame_ssim > self.key_frames_params.metric_threshold
            )[0]
            selected_frames = selected_frames[
                np.where(
                    selected_frames - frame_idx
                    > self.key_frames_params.max_waiting_frames
                )[0]
            ]

            # If there are any indices, find the index of the maximum value
            if selected_frames.size > 0:
                max_idx_frame = self.get_jump_frame(selected_frames)
                key_frames[(frame_idx / FPS)] = max_idx_frame / FPS
                # (Jump_frame, ssim_value)

        return key_frames

    def get_jump_frame(self, selected_frames: np.ndarray) -> int | float:
        # to get the index with the max value
        """max_value_idx = np.argmax(row[indices])
        max_value = row[indices][max_value_idx]
        max_index = indices[max_value_idx]"""
        # to get the bigger idx (the nearest to the final frame)
        max_idx_frame = max(selected_frames)
        return max_idx_frame

    def get_crop_image(
        self, optimal_crop: FaceCoordinates, image: np.ndarray
    ) -> np.ndarray:
        cropped_image = image[
            optimal_crop.top : optimal_crop.bottom,
            optimal_crop.left : optimal_crop.right,
        ]
        return cropped_image

    def get_resized_image(self, image: np.ndarray) -> np.ndarray:
        width = int(image.shape[1] * self.key_frames_params.resize_factor)
        height = int(image.shape[0] * self.key_frames_params.resize_factor)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    def get_crop_frames(self, optimal_crop: FaceCoordinates) -> List[np.ndarray]:
        self.video_reader.start()
        crops_list = []
        for image in self.video_reader:
            image = self.get_crop_image(optimal_crop, image)
            image = self.get_resized_image(image)
            crops_list.append(image)
        return crops_list

    def get_smooth_landmarks(
        self, optimal_crop: FaceCoordinates, window_size: int = 7
    ) -> List[FaceLandmarks]:
        logger.info("Obtaining smoothed face landmarks...")
        face_landmarks_list = self.get_image_landmarks(optimal_crop)
        smoothed_landmarks_list = []
        if window_size % 2 == 0:
            window_size += 1

        half_window = window_size // 2

        for i in range(len(face_landmarks_list)):
            windowed_landmarks = np.asarray(
                [
                    face_landmarks_list[j].to_numpy()
                    for j in range(
                        max(i - half_window, 0),
                        min(i + half_window + 1, len(face_landmarks_list)),
                    )
                ]
            )
            smoothed_landmarks_window = np.mean(windowed_landmarks, axis=0, dtype=int)
            smoothed_landmarks_list.append(
                FaceLandmarks(
                    left_eye=smoothed_landmarks_window[0],
                    right_eye=smoothed_landmarks_window[1],
                    nose=smoothed_landmarks_window[2],
                    mouth_left=smoothed_landmarks_window[3],
                    mouth_right=smoothed_landmarks_window[4],
                )
            )

        return smoothed_landmarks_list

    def get_image_landmarks(self, optimal_crop: FaceCoordinates) -> List[FaceLandmarks]:
        face_landmarks_list = []
        self.video_reader.start()
        for image in self.video_reader:
            cropped_image = self.get_crop_image(optimal_crop, image)
            face_landmarks = self.image_feature_extractor.get_face_landmarks(
                cropped_image
            )
            face_landmarks_list.append(face_landmarks)
        return face_landmarks_list
