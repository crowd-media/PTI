
import sys
import kornia
import numpy as np
import cv2
import torch
from dataloaders.dataloader import BlendDataLoader

from inputs_outpus.blend_input import BlendInput
from params.blend_params import BlendParams

sys.path.append('/home/ubuntu/talking-heads-ai')
# from unith_thai.helpers.feature.io.talking_head_features import TalkingHeadFeatures
from unith_thai.helpers.feature.io.face_coordinates import FaceCoordinates
from unith_thai.helpers.writer.frame_writer import FrameWriter




class BlendPipeline():
    def __init__(
        self, 
        frame_writer: FrameWriter
    ):  
        self.frame_writer = frame_writer

    def process(self, data_loader: BlendDataLoader) -> None:
        print("Process started!")
        for idx, batch in enumerate(data_loader):
            print(f"Processing batch {idx}")
            self.process_batch(batch)
        self.frame_writer.stop()

    def process_batch(self, batch: BlendInput) -> None:
        result = self.postprocess_batch(batch)
        self.frame_writer.write_batch(result)

    def postprocess_batch(
        self, inputs: BlendInput
    ) -> np.ndarray:
        torch_inverse_affine_matrices = torch.from_numpy(
            inputs.inverse_affine_matrices.astype(np.float32)
        )
        output = np.transpose(inputs.faces.astype(np.float32), (0,3,1,2))
        torch_output = torch.from_numpy(output)
        batch_height_width = (
            abs(inputs.crop_coordinates[3] - inputs.crop_coordinates[1]),
            abs(inputs.crop_coordinates[2] - inputs.crop_coordinates[0]),
        )
        warped_output = kornia.geometry.transform.warp_affine(
            torch_output,
            torch_inverse_affine_matrices,
            batch_height_width,
            align_corners=False,
        )
        warped_output = warped_output.numpy().astype(np.float32).astype(np.uint8)
        warped_output = np.transpose(warped_output, (0, 2, 3, 1))
        self.blend_view(
            inputs.frames, inputs.crop_coordinates, warped_output, inputs.mask
        )
        return inputs.frames
    
    @staticmethod
    def blend_view(
        frames: np.ndarray,
        coordinates: np.ndarray,
        blend_crop: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        channels = frames.shape[3]
        mask = np.divide(mask, 255.0, dtype="float32")
        mask_img = np.tile(mask, (channels, 1, 1, 1)).transpose(1, 2, 3, 0)
        frames[
            :,
            coordinates[1] : coordinates[3],
            coordinates[0] : coordinates[2],
            :,
        ] = (
            mask_img * blend_crop
            + (1 - mask_img)
            * frames[
                :,
                coordinates[1] : coordinates[3],
                coordinates[0] : coordinates[2],
                :,
            ]
        )