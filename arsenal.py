#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from numba import jit
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA
import numpy.ma as ma
# from models.FastFlowNet_ import FastFlowNet
import cv2


def dotWithTemplatesOpt(tobeDotted, templates):
    results = []
    for template in templates:
        results.append(np.sum(np.inner(tobeDotted, template)/(8*8*2)))
    '''
    results.insert(1, -results[0])
    results.insert(3, -results[2])
    results.insert(5, -results[4])
    results.insert(7, -results[6])
    '''
    # results = results + [-dotProduct for dotProduct in results]
    results = np.array(results)
    results = results * (results > 0)
    return results


def meanOpticalFlow(flow):

    flow = cv2.resize(flow, (64, 64))
    flowX = flow[:, :, 0]
    flowY = flow[:, :, 1]
    #shape = (8, 8)
    #sh = shape[0],flow.shape[0]//shape[0],shape[1],flow.shape[1]//shape[1]
    sh = (8, 8, 8, 8)
    meanFlowX = flowX.reshape(sh).mean(-1).mean(1)
    meanFlowY = flowY.reshape(sh).mean(-1).mean(1)
    meanFlow = np.dstack((meanFlowX, meanFlowY))
    return meanFlow

def evaluate(output, target):

    output = ma.masked_greater(output, 80)
    target = ma.masked_greater(target, 80)
    # print('max', np.max(target))
    # print('min', np.min(target))

    abs_diff = np.abs((output - target))

    mse = float(np.mean((abs_diff)**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(abs_diff))
    lg10 = float(np.mean(np.abs(np.log10(output) - np.log10(target))))
    absrel = float(np.mean(abs_diff / target))

    maxRatio = np.maximum((output/target), (target/output))
    # print(maxRatio)
    delta1 = float(np.mean(maxRatio < 1.25))
    delta2 = float(np.mean(maxRatio < 1.25 ** 2))
    delta3 = float(np.mean(maxRatio < 1.25 ** 3))

    inv_output = 1 / output
    inv_target = 1 / target
    abs_inv_diff = np.abs(inv_output - inv_target)
    irmse = math.sqrt(np.mean((abs_inv_diff)**2))
    imae = float(np.mean(abs_inv_diff))

    return mse, rmse, mae, lg10, absrel, delta1, delta2, delta3, irmse, imae

def makewall(WallDistance0, width0, height0, hg ,K_in):
    translation_O = np.array([0, 0, 0]).reshape(3, 1)
    translation_1 = np.array([[0, 0, 0, 1]])
    WallDistance = WallDistance0
    width = width0
    height = height0
    grid = hg*2

    # K = np.array([[focalLength, 0, width/2, 0],
    #               [0, focalLength, height/2, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    K = K_in
    # print(K.shape)
    K = np.concatenate((K, translation_O), axis=1)
    # print(K.shape)
    # print(translation_1.shape)
    K = np.concatenate((K, translation_1), axis=0)
    
    K_inv = LA.inv(K)

    rotation = [0, 0, 0]
    rx = rotation[0]
    ry = rotation[1]
    rz = rotation[2]
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                  [ np.sin(rz),  np.cos(rz), 0],
                  [          0,    0,          1]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [          0, 1,          0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1,          0,           0],
                  [ 0, np.cos(rx), -np.sin(rx)],
                  [ 0, np.sin(rx),  np.cos(rx)]])

    translation = [0, 0, 0]
    translation = np.array(translation).reshape(3, 1)

    Rt = np.concatenate((Rz @ Ry @ Rx, translation), axis=1)
    Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)

    Rt_inv = LA.inv(Rt)
    
    frameWallPair = np.zeros(height//grid*width//grid*4*2)
    frameWallPair = frameWallPair.reshape(height//grid*width//grid, 4*2)

    for u in range(grid//2, width, grid):
        for v in range(grid//2, height, grid):
            p_frame = np.array([u, v, 1, 1]).reshape(4, 1)
            p_wall = Rt_inv @ K_inv @ p_frame
            #print(p_wall)
            p_wall = p_wall / p_wall[2] * WallDistance
            frameWallPair[u//grid*(height//grid)+v//grid] = np.concatenate((p_frame, p_wall), axis=None)

    return frameWallPair

@jit(cache=True)
def xyz(wall, K, Rt_translate,width,height, hg):
    hg = hg
    arrayidealVectorT = np.zeros((width,height,2))
    idealMag = np.zeros((width,height))
    for point in wall:	 
        x, y = int(point[0]), int(point[1])
        [ut, vt, wt] = K @ Rt_translate @ np.array([point[4], point[5], point[6],1])
        
        ut = ut/(wt)
        vt = vt/(wt)
        wt = 1.0	
        idealVector = np.array([ut-x, vt-y]).astype(np.float32)
        mag = LA.norm(idealVector)

        for i in range(y-hg, y+hg):
            for j in range(x-hg, x+hg):
                arrayidealVectorT[i,j,0] = ut-x
                arrayidealVectorT[i,j,1] = vt-y
                idealMag[i,j] = mag

    return idealMag, arrayidealVectorT

@jit(cache=True)
def RYP(wall, K, Rt_rotate,width,height, hg):
    hg = hg
    arrayidealVectorRT = np.zeros((width,height,2))
    idealMag = np.zeros((width,height))
    for point in wall:	
        x, y = int(point[0]), int(point[1]) 
        [ur, vr, wr] = K @ Rt_rotate @ np.array([point[4], point[5], point[6],1])
        
        ur = ur/(wr)
        vr = vr/(wr)
        wr = 1.0
        idealVector = np.array([ur-x, vr-y]).astype(np.float32)
        mag = LA.norm(idealVector)
        
        for i in range(y-hg, y+hg):
            for j in range(x-hg, x+hg):
                arrayidealVectorRT[i,j,0] = ur-x
                arrayidealVectorRT[i,j,1] = vr-y
                idealMag[i,j] = mag

    return idealMag, arrayidealVectorRT

def rotationinradius(crx,cry,crz):

    # Rz = np.array([ [ np.cos(crz), -np.sin(crz), 0],
    #                 [ np.sin(crz),  np.cos(crz), 0],
    #                 [       0,      0,      1       ]])
    # Ry = np.array([[np.cos(cry), 0, np.sin(cry)],
    #                 [          0,      1,               0],
    #                 [-np.sin(cry), 0, np.cos(cry)]])
    # Rx = np.array([[1,           0,            0],
    #                 [ 0, np.cos(crx), -np.sin(crx)],
    #                 [ 0, np.sin(crx),  np.cos(crx)]])
    rzyx = R.from_euler('zyx', [crz, cry, crx], degrees=False).as_matrix()
    RI = np.eye(3, 3)

    return rzyx, RI

def translationinmeters(ctx,cty,ctz,crx,cry,crz):
    
    translation = np.array([ctx, cty, ctz]).reshape(3, 1)
    translation = R.from_euler('zyx', [crz, cry, crx], degrees=False).as_matrix() @ translation
    translation_O = [0, 0, 0]
    translation_O = np.array(translation_O).reshape(3, 1)

    return translation, translation_O


@jit(cache=True)
def depthmodule(wall, hg, depth, walldistance, ITF, NRF, diff, tz):
    newSNRF = np.zeros_like(depth)
    for point in wall:
        x, y = int(point[0]), int(point[1])
        hg = hg
        for i in range(y-hg,y+hg):
            for j in range(x-hg,x+hg):
                if(diff[i,j] >= 0 and ITF[i,j] >= 0.05):
                    depth[i,j] = ITF[i,j] / ((np.sqrt(NRF[i,j,0]**2+NRF[i,j,1]**2))) * (walldistance + tz)
                    newSNRF[i,j] = np.sqrt(NRF[i,j,0]**2+NRF[i,j,1]**2)
    
    return depth, newSNRF
    
# @jit(cache=True)
# def depthmodule(wall, hg, depth, walldistance, ITF, NRF):
#     for point in wall:
#         x, y = int(point[0]), int(point[1])
#         hg = 16//2
#         for i in range(y-hg,y+hg):
#             for j in range(x-hg,x+hg):
#                 if(ITF[i,j] >= 0):
#                     depth[i,j] = ((np.sqrt(NRF[i,j,0]**2+NRF[i,j,1]**2)) - ITF[i,j])
#                     # if depth[i,j] < 0:
#                     #     depth[i,j] = 0
    
#     return depth
"""
def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_mix.pth'))

def Fastflow(img1, img2):
    div_flow = 20.0
    div_size = 64
    img1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img1, img2, _ = centralize(img1, img2)

    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    input_t = torch.cat([img1, img2], 1).cuda()

    output = model(input_t).data

    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    flow = flow[0].cpu().permute(1, 2, 0).numpy()

    return flow
"""
