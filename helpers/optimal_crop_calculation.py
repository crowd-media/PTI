import sys
import math


from unith_thai.helpers.feature.io.face_coordinates import FaceCoordinates


def extend_crop_margins(optimal_crop:FaceCoordinates, factor:float)->FaceCoordinates:
    left = optimal_crop.left
    right = optimal_crop.right
    top = optimal_crop.top
    bottom = optimal_crop.bottom

    h_center = left + math.floor((right-left)/2)
    v_center = top + math.floor((bottom-top)/2)

    new_right = h_center + math.floor((right-h_center)*factor)
    new_left = h_center - math.floor((h_center-left)*factor)

    new_top = v_center - math.floor((v_center-top)*factor)
    new_bottom = v_center + math.floor((bottom-v_center)*factor)

    return FaceCoordinates(left = new_left, right = new_right, top = new_top, bottom = new_bottom)
