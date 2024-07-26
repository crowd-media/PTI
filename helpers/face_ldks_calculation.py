import numpy as np
import sys
import cv2 


from unith_thai.helpers.detector.dlib_face_detector import DLibFaceDetector
from unith_thai.helpers.feature.image_feature_extractor import ImageFeatureExtractor

processed_image_path= "/home/ubuntu/PTI/images_original/processed/0001.jpeg"
processed_image = cv2.imread(processed_image_path)

landmarks_model_path = "/home/ubuntu/efs/data/models/dlib/align.dat"
face_detector = DLibFaceDetector(landmarks_model_path)

extractor = ImageFeatureExtractor(face_detector, None)
landmarks = extractor.get_face_landmarks(processed_image)

