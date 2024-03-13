import numpy as np
import os 
import glob
import cv2

class ImageLoader:      
    @staticmethod
    def load(faces_path: str) -> np.ndarray:
        if not os.path.exists(faces_path):
            raise FileNotFoundError("Wrong path, cannot load params")
        
        current_directory = os.getcwd()

        os.chdir(faces_path)
        face_list = sorted(glob.glob(f'*.png'))
        os.chdir(current_directory)

        faces=[]
        for face in face_list:
            faces.append(cv2.imread(f'{faces_path}/{face}'))

        faces=np.array(faces)
        return faces

