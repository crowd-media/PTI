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


class Synthesizer:
    def __init__(
        self,
        video_path: str, 
        features: TalkingHeadFeatures, 
        result_path: str,
        duration: int,
        intensity: int, 
        use_last_w_pivots: bool,
        video_reader: StreamVideoReader,
        use_multi_id_training: bool = False   
    ):
        self.video_path = video_path
        self.features = features
        self.result_path = result_path
        self.duration = duration
        self.intensity = intensity
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
                
    def invert_images(self):
        paths_config.input_data_path = f'preprocessed_images/{paths_config.input_data_id}'
        model_id = run_PTI(use_wandb=False, use_multi_id_training=self.use_multi_id_training)
        return model_id
    
    def load_generator(self, model_id):
        if self.use_multi_id_training:
            with open(f'{paths_config.checkpoints_dir}/model_{model_id}_multi_id.pt', 'rb') as f_new: 
                new_G = torch.load(f_new).cuda()
        else:
            with open(f'{paths_config.checkpoints_dir}/model_{model_id}_0000.pt', 'rb') as f_new: 
                new_G = torch.load(f_new).cuda()
        return new_G
    
    def generate_video(self, generator):
        embedding_dir = os.path.join(
            paths_config.embedding_base_dir,
            paths_config.input_data_id,
            paths_config.pti_results_keyword
        )

        smoother = Smoother(embedding_dir, self.duration)
        embedding_dir = smoother.smooth()


        embedding_list = self.read_embeddings(embedding_dir)

        forw_array = np.linspace(0,self.intensity*0.8, num=int(self.duration/4), endpoint=True)
        forw_array_bis = np.linspace(self.intensity*0.8,self.intensity, num=int(self.duration/4), endpoint=True)
        back_array = np.linspace(self.intensity,self.intensity*0.8, num=int(self.duration/4), endpoint=True)
        back_array_bis = np.linspace(self.intensity*0.8,0, num=int(self.duration/4), endpoint=True)
        array = np.concatenate((forw_array, forw_array_bis, back_array, back_array_bis))

        self.interface_gan(embedding_dir, embedding_list, array, generator)

    def interface_gan(self, embedding_dir, embedding_list, array, new_G):
        result_path = f'{self.result_path}/{paths_config.input_data_id}/{self.duration}_{self.intensity}'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for name, coef in zip(embedding_list, array):
            w_pivot = torch.load(f'{embedding_dir}/{name}/0.pt')
            latent_editor = LatentEditorWrapper()
            latents_after_edit = latent_editor.get_single_interface_gan_edits(w_pivot, [coef])

            for direction, factor_and_edit in latents_after_edit.items():
                print(f'Showing {direction} change')
                for latent in factor_and_edit.values():
                    new_image = new_G.synthesis(latent, noise_mode='const', force_fp32 = True)
                    self.plot_syn_images(new_image, f'{name}')
        
    def plot_syn_images(self, img, img_name): 
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
        # np.save(f'{img_name}.npy', img)
        plt.axis('off') 
        resized_image = Image.fromarray(img,mode='RGB') 
        result_path = f'{self.result_path}/{paths_config.input_data_id}/{self.duration}_{self.intensity}'
        resized_image.save(f'{result_path}/{img_name}.png')
        del img 
        del resized_image 
        torch.cuda.empty_cache()

    def read_embeddings(self, embedding_dir):
        current_directory = os.getcwd()
        os.chdir(embedding_dir)
        embedding_list = sorted(glob.glob(f'*'))
        os.chdir(current_directory)
        return embedding_list
        

    def synthesize(self):
        self.extract_images(self.duration)
        model_id = self.invert_images()  
        # model_id = 'WMJZBYILBXEN' #JO
        # model_id = 'WYALONPVCBNU' #AITANA
        # model_id = "YYJTWPVGCLHI" #AITANA 174
        # model_id = "EQYNOWJBVRLF" #AITANA 0000 usar use_multi_id_training": false
        # model_id = "KODUBCLHQJLU" #JO 0000 usar use_multi_id_training": false
        # model_id = "QYTYHBQTOIPF" #RAKAN
        generator = self.load_generator(model_id)  
        # As it is now, the generator is trained to synthesize from a set of identities (multi_id)
        # We've used it training it for the first image and sythesizing image by image from this training
        # TODO: separate invert_images, from training generator, from synthesis
        self.generate_video(generator)

        # delete temporal files
        try:
            shutil.rmtree(f'preprocessed_images/{paths_config.input_data_id}')
            print("Directory 'processed_images' successfully deleted.")
        except OSError as e:
            print(f"Error: 'processed_images': {e.strerror}")

        
        