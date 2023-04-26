import numpy as np
import io
import struct
import serial
from PIL import Image as PILImage
import cv2
import arsenal
from make_Template import make_Ideal_RF, make_Ideal_TF, make_IRF
from timeit import default_timer as timer
# from scipy.special import softmax
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import time
# Camera object to create the snaps/frames/images that
#  will be deserialized later in the opencv code
import sys
sys.path.append("/home/Sakura/.pyenv/versions/3.8.6/lib/python3.8/site-packages/")
import pyopengv


def extract_features_and_keypoints(image):
    # 使用SIFT或ORB等特徵提取器提取特徵
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    # 使用暴力匹配或FLANN匹配器進行特徵匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 應用比率測試（Lowe's ratio test）以選擇優秀的匹配對
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def keypoints_to_points(keypoints1, keypoints2, matches):
    points1 = []
    points2 = []
    for match in matches:
        points1.append(keypoints1[match.queryIdx].pt)
        points2.append(keypoints2[match.trainIdx].pt)
    return np.array(points1), np.array(points2)

def normalized_bearing_vectors(points, camera_matrix):
    # 將圖像點投影到相機坐標系
    points_normalized = cv2.undistortPoints(points.reshape(-1, 1, 2), camera_matrix, None)
    # 讓 z = 1
    points_normalized = np.squeeze(points_normalized).reshape(-1, 2)
    return np.hstack((points_normalized, np.ones((points_normalized.shape[0], 1))))

# only opencv
def estimate_relative_pose(kp1, kp2, matches, K):
    src_pts = kp1.reshape(-1, 1, 2)
    dst_pts = kp2.reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)

    return R, t

class Camera:

    def __init__(self, device='/dev/ttyACM0'):
        """Reads images from OpenMV Cam
        Args:
            device (str): Serial device
        Raises:
            serial.SerialException
        """
        self.port = serial.Serial(device, baudrate=115200,
                                  bytesize=serial.EIGHTBITS,
                                  parity=serial.PARITY_NONE,
                                  xonxoff=False, rtscts=False,
                                  stopbits=serial.STOPBITS_ONE,
                                  timeout=None, dsrdtr=True)

        # Important: reset buffers for reliabile restarts of OpenMV Cam
        self.port.reset_input_buffer()
        self.port.reset_output_buffer()

    def read_image(self):
        """Read image from OpenMV Cam
        Returns:
            image (ndarray): Image
        Raises:
            serial.SerialException
        """

        # Sending 'snap' command causes camera to take snapshot
        self.port.write('snap'.encode())
        self.port.flush()

        # Read 'size' bytes from serial port
        size = struct.unpack('<L', self.port.read(4))[0]
        image_data = self.port.read(size)
        
        image = np.array(PILImage.open(io.BytesIO(image_data)))

        return image

wid = 128
hei = 128
walldistance = 5
FL = 107.57
hg = 16//2
RI = np.eye(3, 3)
K_03 = np.array([[FL, 0, wid//2 ],
			    [0, FL, hei//2 ],
			    [0, 0, 1, ]])  
# K_03 = np.array([[619.13152082,   0,          61.6084372],
# 			     [  0,         715.72291161,  69.46021523],
# 			     [0, 0, 1, ]])  
print(K_03)
dist_coeffs = np.array([[-1.23001715e+01, -1.67838676e+03, -7.94323900e-02,  3.76330876e-02,  8.30444398e+04]])
wall = arsenal.makewall(walldistance, wid, hei, hg, K_03)

hsv = np.zeros((hei, wid, 3)).astype(np.float32)
hsv[...,1] = 255

hsv0 = np.zeros((hei, wid, 3)).astype(np.float32)
hsv0[...,1] = 255

hsv1 = np.zeros((hei, wid, 3)).astype(np.float32)
hsv1[...,1] = 255

translation_O = np.array([0, 0, 0]).reshape(3, 1)
AV = 0.25 #perframe
T = 2

IdealFlowList = []
DIdealFlowList = []
Ppitch = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, AV, 0, 0)
Pyaw = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, AV, 0)
Proll = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, 0, AV)

# Npitch = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, -AV, 0, 0)
# Nyaw = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, -AV, 0)
# Nroll = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, 0, 0, -AV)
Npitch = -Ppitch
Nyaw = -Pyaw
Nroll = -Proll

IdealFlowList.append(Proll)
IdealFlowList.append(Nroll)
IdealFlowList.append(Ppitch)
IdealFlowList.append(Npitch)
IdealFlowList.append(Pyaw)
IdealFlowList.append(Nyaw)
for i in IdealFlowList:
    new = arsenal.meanOpticalFlow(i)
    DIdealFlowList.append(new.flatten())

IdealTFlowList = []
DIdealTFlowList = []
IdealTFlowListname = []
Px = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T, 0, 0)
Py = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, T, 0)
Pz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, 0, -T)
# PzT = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, 0, -T)/1.6
# Pxy = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, T/1.6, 0)
# PxNy = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, -T/1.6, 0)
# Pxz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, 0, -T/1.6)
# PxNz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, T/1.6, 0, 0) - PzT
# Pyz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, T/1.6, -T/1.6)
# PyNz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, T/1.6, 0) - PzT

# Nx = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, -T, 0, 0)
# Ny = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, -T, 0)
# Nz = make_Ideal_TF(wall, K_03, wid, hei, hg, RI, 0, 0, T)
Nx = -Px
Ny = -Py
Nz = -Pz
# Nxy = -Pxy
# Nxz = -Pxz
# Nyz = -Pyz
# NxPy = -PxNy
# NxPz = -PxNz
# NyPz = -PyNz

IdealTFlowList.append(Px)
IdealTFlowList.append(Nx)
IdealTFlowList.append(Py)
IdealTFlowList.append(Ny)
IdealTFlowList.append(Pz)
IdealTFlowList.append(Nz)

# IdealTFlowList.append(Pxy)
# IdealTFlowList.append(Nxy)
# IdealTFlowList.append(PxNy)
# IdealTFlowList.append(NxPy)
# IdealTFlowList.append(Pxz)
# IdealTFlowList.append(Nxz)
# IdealTFlowList.append(PxNz)
# IdealTFlowList.append(NxPz)
# IdealTFlowList.append(Pyz)
# IdealTFlowList.append(Nyz)
# IdealTFlowList.append(PyNz)
# IdealTFlowList.append(NyPz)

IdealTFlowListname.append('Px')
IdealTFlowListname.append('Nx')
IdealTFlowListname.append('Py')
IdealTFlowListname.append('Ny')
IdealTFlowListname.append('Pz')
IdealTFlowListname.append('Nz')

# IdealTFlowListname.append('Pxy')
# IdealTFlowListname.append('Nxy')
# IdealTFlowListname.append('PxNy')
# IdealTFlowListname.append('NxPy')
# IdealTFlowListname.append('Pxz')
# IdealTFlowListname.append('Nxz')
# IdealTFlowListname.append('PxNz')
# IdealTFlowListname.append('NxPz')
# IdealTFlowListname.append('Pyz')
# IdealTFlowListname.append('Nyz')
# IdealTFlowListname.append('PyNz')
# IdealTFlowListname.append('NyPz')

for i in IdealTFlowList:
    new = arsenal.meanOpticalFlow(i)
    DIdealTFlowList.append(new.flatten())
    # print(len(DIdealFlowList))
    # print(new.shape)

currentFrame = 0
dis = cv2.DISOpticalFlow_create(0)
dis.setFinestScale(1)
start_point = np.array([0, 0, 0]).astype(np.float64)
All_points = []

# Create sample 3D points (replace these with your 3D model points)
points = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [20, 0, 0],
    [0, 20, 0],
    [0, 0, 20],
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10],
    [5, 0, 0],
    [0, 5, 0],
    [0, 0, 5],
    [30, 0, 0],
    [0, 30, 0],
    [0, 0, 30],
    [25, 0, 0],
    [0, 25, 0],
    [0, 0, 25],
    [15, 0, 0],
    [0, 15, 0],
    [0, 0, 15],
    [-20, 0, 0],
    [0, -20, 0],
    [0, 0, -20],
    [-10, 0, 0],
    [0, -10, 0],
    [0, 0, -10],
    [-5, 0, 0],
    [0, -5, 0],
    [0, 0, -5],
    [-30, 0, 0],
    [0, -30, 0],
    [0, 0, -30],
    [-25, 0, 0],
    [0, -25, 0],
    [0, 0, -25],
    [-15, 0, 0],
    [0, -15, 0],
    [0, 0, -15],
])

# grid = pv.UniformGrid()
# grid.points = points
polydata = pv.PolyData(points)

colors = np.array([
    [2, 2, 2, 1],
    [0, 0, 0, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
    [0.9, 0.9, 0.9, 1],
    [1.3, 1.4, 1.5, 1],
    [0.1, 0.3, 0.5, 1],
])

# grid.point_arrays['colors'] = colors
polydata.point_data['colors'] = colors
# Create a pyvista.PolyData object from the points
# polydata = pv.PolyData(points)

# Create a plotter and add the points
plotter = BackgroundPlotter()
plotter.add_mesh(polydata, scalars='colors', point_size=10, render_points_as_spheres=True)

# Add axes for reference
# Add axes for reference
plotter.add_axes()

# Set initial camera position
camera_position = [
    (50, 50, 50),  # Camera location
    (0, 0, 0),  # Focal point
    (0, -1, 0)   # View up direction
]
plotter.camera_position = camera_position
detector = cv2.SIFT_create()
IRF = make_Ideal_RF(wall, K_03, wid, hei, hg, translation_O, AV, 0, 0)
RR_total = np.eye(3)
TT_total = np.zeros((3, 1))
while(True):
    Vectorimage = np.zeros((320, 320, 3), np.uint8)
    # Create a camera by just giving the ttyACM depending on your connection value
    # Change the following line depending on your connection
    strat = timer()
    cap = Camera(device='/dev/ttyACM0')
    # Capture frame-by-frame
    im1 = cap.read_image()
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im1 = cv2.cvtColor(np.float32(im1), cv2.COLOR_GRAY2BGR)
    # im1 = cv2.undistort(im1, K_03, dist_coeffs, None, K_03)
    # im1 = cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    if currentFrame == 0:
        im0 = im1
        keypoints0, descriptors0 = extract_features_and_keypoints(im0)

    keypoints1, descriptors1 = extract_features_and_keypoints(im1)
    
    if len(keypoints0) > 20 and len(keypoints1) > 20 :
        matches = match_features(descriptors0, descriptors1)
        # points0 = keypoints_to_points(keypoints0, matches)
        # points1 = keypoints_to_points(keypoints1, matches)
        points0, points1 = keypoints_to_points(keypoints0, keypoints1, matches)
        # 使用相對姿態RANSAC旋轉估計相機旋轉矩陣
        # bearing_vectors0 = normalized_bearing_vectors(points0, K_03)
        # bearing_vectors1 = normalized_bearing_vectors(points1, K_03)
        # rotation = pyopengv.relative_pose_ransac_rotation_only(bearing_vectors0.astype(np.float64), bearing_vectors1.astype(np.float64), 1e-6)

        RR, TT = estimate_relative_pose(points0, points1, matches, K_03)
        RR_total = RR.dot(RR_total)
        # TT = RR.dot(TT)
        # TT[TT>0.8] = 0
        # TT[TT<-0.8] = 0
        # TT = np.reshape(TT, (3))
        # print(TT.shape)
        # matches = match_features(descriptors0, descriptors1)
        # rotation_matrix = compute_relative_rotation(keypoints0, keypoints1, matches)

        # print('Rota')
        # print(np.round(rotation_matrix, 3))
        # IRF = -make_IRF(wall, K_03, wid, hei, hg, translation_O, rotation)
        IRF = make_IRF(wall, K_03, wid, hei, hg, translation_O, RR)
    else:
        IRF = np.zeros((128,128,2))
    
    # flow = dis.calc(im0, im1, None, )
    flow = cv2.calcOpticalFlowFarneback(im0, im1, None, 0.5, 8, 15, 3, 5, 1.2, 0)
    ori_flow = flow
    flow = flow - IRF
    Dflow = arsenal.meanOpticalFlow(flow)
    DotResult = arsenal.dotWithTemplatesOpt(Dflow.flatten(), DIdealTFlowList)
    
    DotResult[DotResult<50] = 0
    GraphArray = np.round(DotResult/np.sum(DotResult), 3)
    # print(np.round(DotResult, 3))
    if np.mean(DotResult) > 5:
        # DotResult = softmax(DotResult).tolist()
        # print(np.round(DotResult/np.sum(DotResult), 1))
        end_point = np.array([GraphArray[0] - GraphArray[1], GraphArray[3] - GraphArray[2], GraphArray[4] - GraphArray[5]])
        # end_point = np.array([TT[0], TT[1], TT[2]])
        # print('vectors', np.round(end_point, 3))
        
        start_point += end_point
        # print('positions', np.round(start_point, 3))
    
        
        dist_coef = np.zeros((4, 1))
        KK = np.array([[150, 0, 160], [0, 150, 160], [0, 0, 1]]).astype(np.float32)
        # KK = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
        points, _ = cv2.projectPoints(np.array([start_point]), (0, 0, 0), (0, 0, 0), KK, dist_coef)


        # time.sleep(0.1)

        # start_point_2d = tuple(map(int, points[0][0]))
        # end_point_2d = (160, 160)
        # cv2.line(Vectorimage, start_point_2d, end_point_2d, (255, 255, 255), 2)
        # All_points.append(start_point)
        polydata.points[0] = start_point
        plotter.update()

        #print(np.round(DotResult, 3))
        # print(IdealTFlowListname[DotResult.index(np.max(DotResult))])

    # DotResult[4] *= 3
    # DotResult[5] *= 3
    # print('LEN', len(DotResult))
    # if currentFrame%5 == 0:
    
    # print('%s \n'%(DotResult))

    # print(Dflow.shape)
    # RollP = np.inner(flow, Proll)
    # RollN = np.inner(flow, Nroll)
    # PitchP = np.inner(flow, Ppitch)
    # PitchN = np.inner(flow, Npitch)
    # YawP = np.inner(flow, Pyaw)
    # YawN = np.inner(flow, Nyaw)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    # cv2.imshow('im1',im1)
    # cv2.imshow('im1',gray)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = mag*3
    hsv = hsv.astype(np.uint8)
    # print(hsv.shape)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    bgr = cv2.resize(bgr, (320, 320))

    mag0, ang0 = cv2.cartToPolar(IRF[...,0], IRF[...,1])
    hsv0[...,0] = ang0*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv0[...,2] = mag0*3
    hsv0 = hsv0.astype(np.uint8)
    bgr0 = cv2.cvtColor(hsv0,cv2.COLOR_HSV2BGR)
    bgr0 = cv2.resize(bgr0, (320, 320))

    mag1, ang1 = cv2.cartToPolar(ori_flow[...,0], ori_flow[...,1])
    hsv1[...,0] = ang1*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv1[...,2] = mag1*3
    hsv1 = hsv1.astype(np.uint8)
    bgr1 = cv2.cvtColor(hsv1,cv2.COLOR_HSV2BGR)
    bgr1 = cv2.resize(bgr1, (320, 320))

    IMG = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    IMG = cv2.resize(IMG, (320, 320))
    merge0 = np.concatenate((IMG, bgr), axis=1)
    merge1 = np.concatenate((bgr0, bgr1), axis=1)
    merge = np.concatenate((merge0, merge1), axis=0)
    cv2.imshow('merge', merge)
    # cv2.imshow('sda', Vectorimage)

    end = timer()
    # print('FPS', 1/(end-strat))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1
    im0 = im1
    keypoints0 = keypoints1
    descriptors0 = descriptors1