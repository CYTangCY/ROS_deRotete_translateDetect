import rospy
import message_filters
import math
import cv2
import numpy as np
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import Pose, Quaternion,PoseWithCovarianceStamped, PoseStamped
from cv_bridge import CvBridge
import time
import os
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
from scipy.spatial.transform import Rotation as R

import arsenal
from make_Template import make_Ideal_RF, make_Ideal_TF, make_IRF
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import graph

def NormalizeData(data):
	return (data+0.001 - np.min(data+0.001)) / (np.max(data+0.001) - np.min(data+0.001))

#--------------init setting----------------
bg = CvBridge()
K = np.array([[144.778, 0., 161.234],
			  [0., 144.830, 97.921],
			  [0., 0., 1.]])

D = np.array([0.0043, 0.0160, -0.0825, 0.0701])
R_ItoC = np.array([[ 0.000,  0.259, -0.966],
                   [-1.000,  0.000,  0.000],
                   [ 0.000,  0.966,  0.259]])

focalLength0 = 144.778
dis = cv2.DISOpticalFlow_create(0)
dis.setFinestScale(1)
wid = 320
hei = 200

#--------------init setting----------------

#------------virtual setting---------------
KK = np.array([[150, 0, 160], [0, 150, 160], [0, 0, 1]]).astype(np.float32)
hg = 2//2
RI = np.eye(3, 3)
translation_O = np.array([0, 0, 0]).reshape(3, 1)
walldistance = 3
wall = arsenal.makewall(walldistance, wid, hei, hg, K)
dist_coef = np.zeros((4, 1))
#------------virtual setting---------------

#-----------------Template-----------------
T = 1
IdealTFlowList = []
DIdealTFlowList = []
Px = make_Ideal_TF(wall, K, wid, hei, hg, RI, T, 0, 0)
Py = make_Ideal_TF(wall, K, wid, hei, hg, RI, 0, T, 0)
Pz = make_Ideal_TF(wall, K, wid, hei, hg, RI, 0, 0, T)
Nx = -Px
Ny = -Py
Nz = -Pz
IdealTFlowList.append(Px)
IdealTFlowList.append(Nx)
IdealTFlowList.append(Py)
IdealTFlowList.append(Ny)
IdealTFlowList.append(Pz)
IdealTFlowList.append(Nz)
for i in IdealTFlowList:
    new = arsenal.meanOpticalFlow(i)
    DIdealTFlowList.append(new.flatten())
#-----------------Template-----------------

print('loading_success')

class flowdep_pureIMU:
	def __init__(self):
		rospy.init_node('flowdep_pureIMU', anonymous=True)
		
		self.prev_image = None
		self.td = rospy.Duration.from_sec(-0.2)

		self.depth_pub = rospy.Publisher("flowdep_pureIMU/depth", Image, queue_size=1)
		self.diff_pub = rospy.Publisher("imu_diff/pose", PoseStamped, queue_size=1)	
		self.posi = rospy.Publisher("flowdep_pureIMU/posi", PoseStamped, queue_size=1)

		imu_sub = message_filters.Subscriber("/rtimulib_node/imu", Imu)
		image_sub = message_filters.Subscriber("/cam0/image_raw", Image)

		# 0.5 sec buffer cache
		self.imu_cache = message_filters.Cache(imu_sub, 60)	 # ~119Hz
		self.image_cache = message_filters.Cache(image_sub, 1) # ~60fps


		self.imu_cache.registerCallback(self.imu_callback)
		self.image_cache.registerCallback(self.image_callback)
		
		self.framerate = 0

		self.prev_stamp = None
		self.curr_stamp = None
		self.framecount = 0
		self.prev_orientation = None

		#-----------visualize setting-------------
		self.hsv = np.zeros((hei, wid, 3)).astype(np.float32)
		self.hsv[...,1] = 255
		self.hsv0 = np.zeros((hei, wid, 3)).astype(np.float32)
		self.hsv0[...,1] = 255
		self.hsv1 = np.zeros((hei, wid, 3)).astype(np.float32)
		self.hsv1[...,1] = 255

		self.start_point = np.array([0, 0, 0]).astype(np.float64)
		points = graph.PV_Point()
		colors = graph.PV_Color()
		self.polydata = pv.PolyData(points)
		self.polydata.point_data['colors'] = colors

		# self.my_plotter = MyPlotter()
		# self.plot_init()
		#-----------visualize setting-------------

		rospy.spin()
		
		# print(IMGIMU.shape)

	def update_plotter(self):
		self.polydata.points[0] = self.start_point
		return

	def imu_callback(self, imu):
		# image-based algorithm, so nothing here
		return

	def image_callback(self, image):
		avx = 0
		avy = 0
		avz = 0
		self.framecount += 1
		if self.framecount == 1:
			self.plotter = BackgroundPlotter()
			self.plotter.add_mesh(self.polydata, scalars='colors', point_size=10, render_points_as_spheres=True)
			self.plotter.add_axes()
			self.plotter.camera_position = graph.PV_Camera_position()		

		begin = timer()
		IMU_count = 1
		ObstacleList = []
		# get current time
		self.curr_stamp = self.image_cache.getLatestTime()
		# initialization check
		if (self.prev_stamp is None) or (self.prev_stamp > self.curr_stamp):
			self.prev_stamp = self.curr_stamp
		
		# get image
		cv_image = bg.imgmsg_to_cv2(self.image_cache.getElemBeforeTime(self.curr_stamp))
		bgr_cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
		ud_cv_image = cv2.fisheye.undistortImage(cv_image, K, D, Knew=K)

		for imu_data in self.imu_cache.getInterval(self.prev_stamp+self.td, self.curr_stamp+self.td):

			if IMU_count == 1:
				avx = 0
				avy = 0
				avz = 0
			avx += imu_data.angular_velocity.x
			avy += imu_data.angular_velocity.z
			avz += imu_data.angular_velocity.y	 
			IMU_count += 1

		if self.prev_orientation is None:
			self.prev_orientation = self.imu_cache.getElemAfterTime(self.prev_stamp+self.td).orientation

		imu_data = self.imu_cache.getElemBeforeTime(self.curr_stamp+self.td)

		if imu_data is None:
			return			
		
		if IMU_count > 1:
			avx = avx/IMU_count
			avy = avy/IMU_count
			avz = avz/IMU_count 

		RR_GtoIi = R.from_quat([imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w])
		R_GtoIi = RR_GtoIi.as_matrix()
		#R_GtoC = R_ItoC @ R_GtoIi
		R_GtoC =  R_GtoIi @ R_ItoC

		prev_RR_GtoIi = R.from_quat([self.prev_orientation.x, self.prev_orientation.y, self.prev_orientation.z, self.prev_orientation.w])
		prev_R_GtoIi = prev_RR_GtoIi.as_matrix()
		#prev_R_GtoC = R_ItoC @ prev_R_GtoIi
		prev_R_GtoC = prev_R_GtoIi @ R_ItoC
		#R_dCinC = R_GtoC @ np.linalg.inv(prev_R_GtoC)
		R_dCinC = np.linalg.inv(prev_R_GtoC) @ R_GtoC
		q_dCinC = R.from_matrix(R_dCinC).as_quat()
		q_dCinC[0] *= -1
		R_dCinC = R.from_quat(q_dCinC).as_matrix()

		IMU_timestep = (self.curr_stamp - self.prev_stamp).to_sec()

		#-----------IRF-----------------
		IRF = make_Ideal_RF(wall, K, wid, hei, hg, translation_O, avx*IMU_timestep*2, -avy*IMU_timestep*2, -avz*IMU_timestep*2)
		# IRF = make_IRF(wall, K, wid, hei, hg, translation_O, R_dCinC)
		#-----------IRF-----------------

		if self.prev_image is None:
			self.prev_image = ud_cv_image.copy()


		flow = dis.calc(self.prev_image, ud_cv_image, None, )
		# flow = cv2.calcOpticalFlowFarneback(self.prev_image, ud_cv_image, None, 0.5, 8, 15, 3, 5, 1.2, 0)
		# diff = cv2.absdiff(self.prev_image, ud_cv_image) 

		ori_flow = flow

		#------------------De-Rotate----------------------
		flow = flow - IRF
		# print(np.mean(IRF))
		#------------------De-Rotate----------------------

		Dflow = arsenal.meanOpticalFlow(flow)
		DotResult = arsenal.dotWithTemplatesOpt(Dflow.flatten(), DIdealTFlowList)
		switch = DotResult[5]
		DotResult[5] = DotResult[4]
		DotResult[4] = switch
		# DotResult[DotResult[0]<100, ...] = 0
		# DotResult[DotResult[1]<100, ...] = 0

		DotResult[DotResult<50] = 0

		GraphArray = np.round((DotResult+0.0001)/np.sum(DotResult+0.0001), 3)	
		
		
		stamp = rospy.Time.now()        
		# publish rotation in camera frame
		p = PoseStamped()
		p.header.stamp = stamp
		p.header.frame_id = "imu"
		p.pose.position.x = 0
		p.pose.position.y = 0
		p.pose.position.z = 0
		p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = R.from_matrix(R_dCinC).as_quat()
		self.diff_pub.publish(p)		

		#--------------visualization--------------
		# if np.mean(DotResult) > 5:
		end_point = np.array([GraphArray[0] - GraphArray[1], GraphArray[3] - GraphArray[2], GraphArray[4] - GraphArray[5]])
		self.start_point += end_point
		self.update_plotter()
		
		# self.my_plotter.update_plot(self.start_point)
		
		# print('55688', self.start_point)
		# print('dasdasdasd', self.start_point)
		# points, _ = cv2.projectPoints(np.array([self.start_point]), (0, 0, 0), (0, 0, 0), KK, dist_coef)

		pp = PoseStamped()
		pp.header.stamp = stamp
		pp.header.frame_id = "imu"
		pp.pose.position.x = 0
		pp.pose.position.y = 0
		pp.pose.position.z = 0	
		pp.pose.position.x = self.start_point[0]
		pp.pose.position.y = self.start_point[1]
		pp.pose.position.z = self.start_point[2]		
		self.posi.publish(pp)	
		

		mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
		self.hsv[...,0] = ang*180/np.pi/2
		# self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		self.hsv[...,2] = mag*5
		self.hsv = self.hsv.astype(np.uint8)
		# print(hsv.shape)
		bgr = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)
		# bgr = cv2.resize(bgr, (200, 320))

		mag0, ang0 = cv2.cartToPolar(IRF[...,0], IRF[...,1])
		self.hsv0[...,0] = ang0*180/np.pi/2
		# self.hsv0[...,2] = cv2.normalize(mag0,None,0,255,cv2.NORM_MINMAX)
		self.hsv0[...,2] = mag0*5
		self.hsv0 = self.hsv0.astype(np.uint8)
		bgr0 = cv2.cvtColor(self.hsv0,cv2.COLOR_HSV2BGR)
		# bgr0 = cv2.resize(bgr0, (200, 320))

		mag1, ang1 = cv2.cartToPolar(ori_flow[...,0], ori_flow[...,1])
		self.hsv1[...,0] = ang1*180/np.pi/2
		# self.hsv1[...,2] = cv2.normalize(mag1,None,0,255,cv2.NORM_MINMAX)
		self.hsv1[...,2] = mag1*5
		self.hsv1 = self.hsv1.astype(np.uint8)
		bgr1 = cv2.cvtColor(self.hsv1,cv2.COLOR_HSV2BGR)
		# bgr1 = cv2.resize(bgr1, (200, 320))

		IMG = bgr_cv_image
		# IMG = cv2.resize(IMG, (200, 320))
		merge0 = np.concatenate((IMG, bgr), axis=1)
		merge1 = np.concatenate((bgr0, bgr1), axis=1)
		merge = np.concatenate((merge0, merge1), axis=0)
		# print(merge.shape)
		
		cv2.imshow('merge', merge)
		# print(1)
		end = timer()
		cv2.waitKey(1)
		# print(1)
		#--------------visualization--------------
		self.prev_image = ud_cv_image
		self.prev_stamp = self.curr_stamp
		# self.prev_orientation = imu_data.orientation 
		self.framerate += (1/(end-begin))		
		self.prev_orientation = imu_data.orientation  
		if self.framecount%15 == 0:
			# print('1', DotResult)
			# print('2', GraphArray)
			# print(self.framerate/30)
			self.framerate = 0
		return 
					   

if __name__ == '__main__':
	obj = flowdep_pureIMU()
	obj.image_callback(image)

