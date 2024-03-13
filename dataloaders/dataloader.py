import logging
import math

import numpy as np

from unith_thai.data_loaders.frame_reader import FrameReader
from unith_thai.helpers.feature.io.talking_head_features import TalkingHeadFeatures

from inputs_outpus.blend_input import BlendInput

logger = logging.getLogger(__name__)


class BlendDataLoader:
    def __init__(
        self,
        video_reader: FrameReader,
        talking_head_features: TalkingHeadFeatures,
        faces: np.ndarray,
        batch_size: int,
    ):
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        self.batch_size = batch_size
        self.index = 0
        self.video_reader = video_reader
        self.talking_head_features = talking_head_features
        self.faces = faces
        self.start()

    def next_batch(self) -> BlendInput:
        frames, conversion_params, faces = self.read_batch()
        return BlendInput(
            frames=frames,
            affine_matrices=conversion_params.affine_matrix,
            inverse_affine_matrices=conversion_params.inverse_affine_matrix,
            mask=conversion_params.mask,
            crop_coordinates=conversion_params.crop_coordinates,
            faces=faces
        )

    def read_batch(self) -> (np.ndarray, TalkingHeadFeatures, np.ndarray):
        frames = []
        for _ in range(self.number_of_elements_batch()):
            frames.append(next(self.video_reader))
        index_start = (self.index * self.batch_size) % len(self.video_reader)
        index_end = index_start + self.number_of_elements_batch()
        index_overflow = max(
            self.number_of_elements_batch() - self.number_of_video_elements_batch(), 0
        )
        faces=np.concatenate(
            [
                self.faces[index_start:index_end],
                self.faces[:index_overflow],
            ]
        )
        talking_head_features = TalkingHeadFeatures(
            affine_matrix=np.concatenate(
                [
                    self.talking_head_features.affine_matrix[index_start:index_end],
                    self.talking_head_features.affine_matrix[:index_overflow],
                ]
            ),
            inverse_affine_matrix=np.concatenate(
                [
                    self.talking_head_features.inverse_affine_matrix[
                        index_start:index_end
                    ],
                    self.talking_head_features.inverse_affine_matrix[:index_overflow],
                ]
            ),
            mask=np.concatenate(
                [
                    self.talking_head_features.mask[index_start:index_end],
                    self.talking_head_features.mask[:index_overflow],
                ]
            ),
            crop_coordinates=self.talking_head_features.crop_coordinates,
            key_frames=None,
        )
        return np.asarray(frames), talking_head_features, faces

    def number_of_elements(self) -> int:
        return len(self.faces)

    def number_of_elements_batch(self) -> int:
        remaining_elements = self.number_of_elements() - (self.index * self.batch_size)
        return min(remaining_elements, self.batch_size)

    def number_of_video_elements_batch(self) -> int:
        remaining_elements = len(self.video_reader) - (
            self.index * self.batch_size
        ) % len(self.video_reader)
        return min(remaining_elements, self.batch_size)

    def update_video_index(self, index: int) -> None:
        self.video_reader.update_index(index)

    def start(self) -> None:
        self.video_reader.start()

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return math.ceil(self.number_of_elements() / self.batch_size)

    def __next__(self) -> BlendInput:
        if self.index >= len(self):
            raise StopIteration
        batch = self.next_batch()
        self.index += 1
        return batch
