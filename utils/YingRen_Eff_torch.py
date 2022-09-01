# -*- coding: utf-8 -*-
import cv2
import numpy as np

#import tensorflow as tf
#EfficientDet
import torch
from torch.backends import cudnn
from Effbackbone import EfficientDetBackbone #檔案
from efficientdet.utils import BBoxTransform, ClipBoxes#資料夾
from EFFutils.utils import preprocess, invert_affine, postprocess, preprocess_video#資料夾
#
import sys
sys.path.append("..")
#from utils import label_map_util
import utils.yolov4_label_util as yolo_label
import random
#必要參數


class Eff_D1_YingRen():

    def __init__(self, Eff_Model_path, PRED_NAMES):
        """
        EfficientDet D1 class init setting
        :param Eff_Model_path: the model check point path or frozen model pb file path
        :param PRED_NAMES: the model label text path
        """
        self.names, CLASS_NUM = yolo_label.load_names(PRED_NAMES)
        self.model = EfficientDetBackbone(compound_coef=1, num_classes=CLASS_NUM)
        self.model.load_state_dict(torch.load(f'{Eff_Model_path}.pth'))
        self.model.requires_grad_(False)
        self.model.eval()
        self.use_cuda = True
        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True
        self.input_size = 640

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()

        self.colors = []
        for i in range(len(self.names)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.colors.append((b, g, r))
    
    def Return_Obj(self,preds,imgs):
        boxes=[]
        scores=[]
        classes=[]
        for i in range(len(imgs)):
            for j in range(len(preds[0]['rois'])):
                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                boxes.append([x1, y1, x2, y2])
                obj = self.names[preds[i]['class_ids'][j]]
                classes.append([obj])
                scores.append(float(preds[i]['scores'][j]))
               
        return boxes, scores, classes

    def catchObject(self, img, threshold=0.9):
        """
        The function of catch objects via threshold, which was the minimized confidence score proposed by model.
        :param img: the image from video or webcam frame
        :param threshold: the minimized confidence score proposed by model
        :return: the objects (data type list) every object contained
        [
        label name (data type string),
        bounding box position (data type numpy array)
        ]
        """
        frame = img
        self.threshold = threshold
        self.iou_threshold = 0.5
        ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=self.input_size)
        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)
            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              self.threshold, self.iou_threshold)
        out = invert_affine(framed_metas, out)
        (boxes, scores, classes) = self.Return_Obj(out, ori_imgs)
        objs = []
        for k in range(len(boxes)):
            obj = []
            if scores[k] >= threshold:
                xmin = int(round(boxes[k][0]))
                ymin = int(round(boxes[k][1]))
                xmax = int(round(boxes[k][2]))
                ymax = int(round(boxes[k][3]))
                obj.append(classes[k][0])
                obj.append(np.array([xmin, ymin, xmax, ymax]))
                objs.append(obj)
        return objs
