import os
import glob
import shutil
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from configs import paths_config
from configs import hyperparameters
from scripts.latent_editor_wrapper import LatentEditorWrapper
from scripts.run_pti import run_PTI
from utils.constants import PTI_SIZE
from helpers.smoother import Smoother

from unith_thai.data_loaders.video.stream_video_reader import StreamVideoReader
from unith_thai.helpers.feature.io.talking_head_features import TalkingHeadFeatures


class Trainer:
    def __init__(
        self,
        video_path: str, 
        features: TalkingHeadFeatures, 
        duration: int,
        use_last_w_pivots: bool,
        video_reader: StreamVideoReader,
        use_multi_id_training: bool = False   
    ):
        self.video_path = video_path
        self.features = features
        self.duration = duration
        self.video_reader = video_reader
        self.use_multi_id_training = use_multi_id_training
     
        paths_config.stylegan2_ada_ffhq = '/home/ubuntu/efs/data/models/stylegan/ffhq.pkl'
        paths_config.input_data_id = (video_path.split('/')[-1]).split('.')[0]
        # TODO: control parameters from config not hyperparameters.py script
        hyperparameters.use_last_w_pivots = use_last_w_pivots

    def extract_images(self,duration):
        destination_path = f'preprocessed_images/{paths_config.input_data_id}'
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        coord = self.features.crop_coordinates
        for i, image in enumerate(self.video_reader):
            if i<duration:
                cropped_image = image[coord[1]:coord[3], coord[0]:coord[2], :]
                warped_image = cv2.warpAffine(cropped_image, self.features.affine_matrix[i], PTI_SIZE)
                cv2.imwrite(f'{destination_path}/{i:04d}.png', warped_image)
                
    def invert_images_and_train(self):
        paths_config.input_data_path = f'preprocessed_images/{paths_config.input_data_id}'
        model_id = run_PTI(use_wandb=False, use_multi_id_training=self.use_multi_id_training)
        return model_id
        

    def synthesize(self):
        self.extract_images(self.duration)
        model_id = self.invert_images_and_train()  

        # delete temporal files
        try:
            shutil.rmtree(f'preprocessed_images/{paths_config.input_data_id}')
            print("Directory 'processed_images' successfully deleted.")
        except OSError as e:
            print(f"Error: 'processed_images': {e.strerror}")

        return model_id


        
        