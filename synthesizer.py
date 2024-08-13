import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from configs import paths_config
from scripts.latent_editor_wrapper import LatentEditorWrapper
from helpers.smoother import Smoother


class Synthesizer:
    def __init__(
        self,
        video_path: str,
        result_path: str,
        duration: int,
        intensity: float, 
        model_id: str, 
        use_multi_id_training: bool = False
    ):
        self.video_path = video_path
        self.result_path = result_path
        self.duration = duration
        self.intensity = intensity
        self.model_id = model_id
        self.use_multi_id_training = False
     
        paths_config.stylegan2_ada_ffhq = '/home/ubuntu/efs/data/models/stylegan/ffhq.pkl'
        paths_config.input_data_id = (video_path.split('/')[-1]).split('.')[0]
        # TODO: control parameters from config not hyperparameters.py script
    
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

        forw_array = np.linspace(0,self.intensity*0.5, num=int(self.duration/4), endpoint=True)
        forw_array_bis = np.linspace(self.intensity*0.5,self.intensity, num=int(self.duration/4), endpoint=True)
        back_array = np.linspace(self.intensity,self.intensity*0.5, num=int(self.duration/4), endpoint=True)
        back_array_bis = np.linspace(self.intensity*0.5,0, num=int(self.duration/4), endpoint=True)
        array = np.concatenate((forw_array, forw_array_bis, back_array, back_array_bis))

        self.interface_gan(embedding_dir, embedding_list, array, generator)

    def interface_gan(self, embedding_dir, embedding_list, array, new_G):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

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
        resized_image.save(f'{self.result_path}/{img_name}.png')
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
        generator = self.load_generator(self.model_id)  
        self.generate_video(generator)
        
        