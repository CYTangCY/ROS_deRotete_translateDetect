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

# wid = 128
# hei = 128
# walldistance = 7
# FL = 107.57
# hg = 16//2
# RI = np.eye(3, 3)
# K_03 = np.array([[FL, 0, wid//2 ],
# 			    [0, FL, hei//2 ],
# 			    [0, 0, 1, ]])  
# print(K_03)
# wall = arsenal.makewall(walldistance, FL, wid, hei, hg)

# hsv = np.zeros((hei, wid, 3)).astype(np.float32)
# hsv[...,1] = 255
# translation_O = np.array([0, 0, 0]).reshape(3, 1)
# AV = 0.25 #perframe

# Proll = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, AV, 0, 0)
# Ppitch = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, AV, 0)
# Pyaw = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, 0, AV)
# Nroll = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, -AV, 0, 0)
# Npitch = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, -AV, 0)
# Nyaw = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, 0, -AV)

"""
print(Ppitch.shape)
mag, ang = cv2.cartToPolar(Proll[...,0], Proll[...,1])
hsv[...,0] = ang*180/np.pi/2
# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
hsv[...,2] = mag*5
hsv = hsv.astype(np.uint8)
# print(hsv.shape)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

bgr = cv2.resize(bgr, (640, 640))
cv2.imshow('IRF', bgr)
cv2.waitKey(2000)
"""