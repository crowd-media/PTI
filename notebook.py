import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper
import cv2

#configuration
image_dir_name = 'image'

## If set to true download desired image from given url. If set to False, assumes you have uploaded personal image to
## 'image_original' dir
use_image_online = True
image_name = 'personal_image'
use_multi_id_training = False
global_config.device = 'cuda'
paths_config.e4e = './pretrained_models/e4e_ffhq_encode.pt'
paths_config.input_data_id = image_dir_name
paths_config.input_data_path = f'./{image_dir_name}_processed'
paths_config.stylegan2_ada_ffhq = '/home/ubuntu/efs/data/models/stylegan/ffhq.pkl'
paths_config.checkpoints_dir = './'
paths_config.style_clip_pretrained_mappers = './pretrained_models'
hyperparameters.use_locality_regularization = False

#preprocess image
pre_process_images(f'./{image_dir_name}_original')

#invert image using PTI
os.chdir('.')
model_id = run_PTI(use_wandb=False, use_multi_id_training=True)

def display_alongside_source_image(images): 
    res = np.concatenate([np.array(image) for image in images], axis=1) 
    return Image.fromarray(res) 

def load_generators(model_id, image_name):
  with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
    old_G = pickle.load(f)['G_ema'].cuda()
    
  with open(f'{paths_config.checkpoints_dir}/model_{model_id}_multi_id.pt', 'rb') as f_new: 
    new_G = torch.load(f_new).cuda()

  return old_G, new_G

generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
old_G, new_G = load_generators(model_id, generator_type)

def plot_syn_images(syn_images): 
  for img in syn_images: 
      img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
      plt.axis('off') 
      resized_image = Image.fromarray(img,mode='RGB').resize((256,256)) 
      cv2.imwrite('resized.png', resized_image)
      del img 
      del resized_image 
      torch.cuda.empty_cache()

w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
w_pivot = torch.load(f'{embedding_dir}/0.pt')

old_image = old_G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)
new_image = new_G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)

print('Upper image is the inversion before Pivotal Tuning and the lower image is the product of pivotal tuning')
plot_syn_images([old_image, new_image])