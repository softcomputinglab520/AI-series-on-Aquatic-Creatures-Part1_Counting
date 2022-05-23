#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:54:28 2019

@author: deep
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:13:15 2018

@author: tim
"""

import cv2
import numpy as np

#Focal Length
left_fc_1 = 1257.12964
left_fc_2 = 1257.13641
#---------------------
right_fc_1 = 1261.22205
right_fc_2 = 1262.55092
#---------------------
#Principal point:
left_cc_1 = 335.41894
left_cc_2 = 251.67348
#---------------------
right_cc_1 = 330.03685
right_cc_2 = 256.02895
#---------------------
#Distortion
left_kc_1 = 0.45017
left_kc_2 = 0.36237
left_kc_3 = 0.01010
left_kc_4 = 0.00195
#--------------------
right_kc_1 = 0.48830
right_kc_2 = 0.10612
right_kc_3 = 0.01205
right_kc_4 = -0.00189
#--------------------
#Rotation vector
om_1 = 0.00000
om_2 = 0.00000
om_3 = 0.00000
#--------------------
#Translation vector
T_1 = 0.00000
T_2 = 0.00000
T_3 = 0.00000
#--------------------
#Stereo calibration parameters after loading the individual calibration files

left_camera_matrix = np.array([[left_fc_1, 0, left_cc_1],
                               [0., left_fc_2, left_cc_2],
                               [0., 0., 1.]])
left_distortion = np.array([[ left_kc_1,   left_kc_2,   left_kc_3,   left_kc_4 , 0.00000]])



right_camera_matrix = np.array([[right_fc_1, 0, right_cc_1],
                                [0, right_fc_2, right_cc_2],
                                [0, 0, 1.]])
right_distortion = np.array([[ right_kc_1, right_kc_2, right_kc_3, right_kc_4,  0.00000 ]])

om = np.array([om_1, om_2, om_3]) # 旋轉關系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues變換將om變換為R
T = np.array([T_1, T_2, T_3]) # 平移關系向量

size = (640,480) # 圖像尺寸

# 進行立體更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 計算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
