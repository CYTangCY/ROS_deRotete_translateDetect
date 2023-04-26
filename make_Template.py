import arsenal
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, roll, pitch, yaw): 
    # Rotatez = R.from_euler('zyx', [0.1, 0, 0], degrees=False).as_matrix()
    # if np.mean(roll + pitch + yaw) < 0.1:
    #     Rotatez = np.eye(3)
    rzyx, RI = arsenal.rotationinradius(roll, pitch, yaw)
    # rzyx = rzyx @ Rotatez
    Rt_rotate = np.concatenate((rzyx, translation_O), axis=1)
    ScalarIRF, idealRotateFlow = arsenal.RYP(wall, K_03, Rt_rotate, hei, wid, hg)

    return idealRotateFlow

def make_Ideal_TF(wall, K_03, wid, hei, hg, RI, x, y, z): 

    Rt_translate = np.concatenate((RI, np.array([x, y, z]).reshape(3, 1)), axis=1)
    idealTranslationFlow, ITFF = arsenal.xyz(wall, K_03, Rt_translate, hei, wid, hg)

    return ITFF

def make_IRF(wall, K_03, wid, hei, hg, translation_O, RM): 

    Rt_rotate = np.concatenate((RM, translation_O), axis=1)
    ScalarIRF, idealRotateFlow = arsenal.RYP(wall, K_03, Rt_rotate, hei, wid, hg)

    return idealRotateFlow
