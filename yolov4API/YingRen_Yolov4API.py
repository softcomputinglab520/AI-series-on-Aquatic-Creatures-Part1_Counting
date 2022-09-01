# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:39:29 2020

@author: MH-Lin
"""


import numpy as np
#yolov4
from ctypes import *
import random
import os
import cv2
import time
from yolov4API import darknet #重要
#
import sys
sys.path.append("..")
#from utils import label_map_util
# import utils.yolov4_label_util as yolo_label
import random
#必要參數

#

class Yolov4YingRen():

    def __init__(self, config_file_Path, data_file_path,weights_Path):
        """
        YOLO v4 class init setting
        :param config_file_Path: config file path of the model
        :param data_file_path: data file path of the model
        :param weights_Path: weights file path of the model
        """
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file_Path,
            data_file_path,
            weights_Path,
            batch_size=1
        )
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        self.colors = []
        for i in range(len(self.class_colors)):
            r = self.class_colors[self.class_names[i]][0]
            g = self.class_colors[self.class_names[i]][1]
            b = self.class_colors[self.class_names[i]][2]
            self.colors.append((b, g, r))

    def catchObject(self, img, threshold=0.9):
        """
        The function of catch objects via threshold, which was the minimized confidence score proposed by model.
        :param img: the image from video or webcam frame
        :param threshold: the minimized confidence score proposed by model
        :return: the image(data type numpy array), the objects (data type list) every object contained
        [
        label name (data type string),
        bounding box position (data type numpy array)
        ]
        """
        if type(img) == str:
            image = img.copy()
        elif type(img) == np.ndarray:
            image = img.copy()
        else:
            print('錯誤，Please input image or image path.')

        frame = image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#轉成RGB
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height),interpolation=cv2.INTER_LINEAR)        
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())#影像轉成darknet專屬格式
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, threshold)
        objs=[]
        w=img.shape[1]
        h=img.shape[0]
        for label, confidence, bbox in detections:
            obj = []
            left, top, right, bottom = darknet.bbox2points(bbox)
            xmin = int((left/self.width) * w)
            ymin = int((top/self.height) * h)
            xmax = int((right/self.width) * w)
            ymax = int((bottom/self.height) * h)
            obj.append(label)
            obj.append(np.array([xmin, ymin, xmax, ymax]))
            objs.append(obj)
        return objs
