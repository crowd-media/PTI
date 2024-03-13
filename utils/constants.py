import numpy as np
import sys

sys.path.append('/home/ubuntu/talking-heads-ai')
from unith_thai.helpers.feature.io.face_landmarks import FaceLandmarks


# face landmarks referred to a 1024 x 1024 px image
PTI_FACE_TEMPLATE = FaceLandmarks(
    left_eye=np.array([383, 485]), 
    right_eye=np.array([639, 483]), 
    nose=np.array([502, 612]), 
    mouth_left=np.array([407, 753]), 
    mouth_right=np.array([619, 745])
    )

PTI_SIZE = (1024, 1024)