import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
from utils import label_map_util
# from utils import visualization_utils as vis_util
import utils.yolov4_label_util as yolo_label
import random
# from ImageFilter import *
# from utils import color_map

class Faster_RCNN():

    def __init__(self, CKPTPATH, LABELPATH):
        PATH_TO_CKPT = CKPTPATH
        PATH_TO_LABELS = LABELPATH
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        NUM_CLASSES = len(label_map.item)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        self.colors = []
        for i in range(NUM_CLASSES):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.colors.append((b, g, r))
        # self.IF = ImageFilter()
        # self.IF.rL = 1.5
        # self.IF.rH = 1

    def catchObject(self, img, threshold=0.9):
        if type(img) == str:
            frame = cv2.imread(img)
        elif type(img) == np.ndarray:
            frame = img.copy()
        else:
            print('Please input image or image path.')
        # frame1=frame.copy()
        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        objs = []
        for k in range(int(scores.size)):
            obj = []
            if scores[0][k] > threshold:
                xmin=int(round(boxes[0][k, 1]*frame.shape[1]))
                ymin=int(round(boxes[0][k, 0]*frame.shape[0]))
                xmax=int(round(boxes[0][k, 3]*frame.shape[1]))
                ymax=int(round(boxes[0][k, 2]*frame.shape[0]))
                obj.append(self.category_index.get(classes[0][k]).get('name').lower())
                obj.append(np.array([xmin, ymin, xmax, ymax]))
                objs.append(obj)
        return objs

    def catchObject01(self,img, threshold=0.9):
        if type(img) == str:
            frame = cv2.imread(img)
        elif type(img) == np.ndarray:
            frame = img.copy()
        else:
            print('Please input image or image path.')
        # frame1=frame.copy()
        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        objs=[]
        for k in range(int(scores.size)):
            obj = []
            if scores[0][k]>threshold:
                xmin=int(round(boxes[0][k, 1]*frame.shape[1]))
                ymin=int(round(boxes[0][k, 0]*frame.shape[0]))
                xmax=int(round(boxes[0][k, 3]*frame.shape[1]))
                ymax=int(round(boxes[0][k, 2]*frame.shape[0]))
                obj.append(self.category_index.get(classes[0][k]).get('name').lower())
                obj.append(np.array([xmin, ymin, xmax, ymax]))
                objs.append(obj)
        return frame, objs

    def catchObject_J(self, img, min_score_thresh, frame_count,visualize_boxes_IO = "OFF"):
        if type(img) == str:  # 給自串路徑
            frame = cv2.imread(img)
            image = frame.copy()
        elif type(img) == np.ndarray:  # 已經是圖片
            frame = img.copy()
            image = frame.copy()
        else:
            print('錯誤，Please input image or image path.')
        frame_expanded = np.expand_dims(frame, axis=0)  # 調整輸入frame的維度
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        objs = []
        for k in range(int(scores.size)):
            obj = []  # 每次清空
            if round(scores[0][k], 5) > min_score_thresh:
                xmin = int(round(boxes[0][k, 1] * frame.shape[1]))
                ymin = int(round(boxes[0][k, 0] * frame.shape[0]))
                xmax = int(round(boxes[0][k, 3] * frame.shape[1]))
                ymax = int(round(boxes[0][k, 2] * frame.shape[0]))
                cx = int(round((xmax+xmin)/2))
                cy = int(round((ymax+ymin)/2))
                obj.append(self.category_index.get(classes[0][k]).get('name').lower())
                obj.append([[xmin, ymin, xmax, ymax]])
                obj.append([[xmax - xmin, ymax - ymin]])
                obj.append([[cx,cy]])
                obj.append(frame_count)
                obj.append(scores[0][k])
                objs.append(obj)
        if visualize_boxes_IO == 'ON':  # 判斷FastRCNN內建可試化視窗開啟或關閉
            for i, obj in enumerate(objs):
                xmin, ymin, xmax, ymax = obj[1][0]
                for ID, data in self.category_index.items():
                    if obj[0] == data['name'].lower():
                        id = ID - 1
#                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.colors[id], 2)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(image, obj[0] + '{:2d}%'.format(int(obj[5] * 100)),
                            (xmin, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[id], 2, cv2.LINE_AA)
        return image, objs

    # def catchObject_Homo(self,img, threshold=0.9):
    #     if type(img) == str:
    #         frame1 = cv2.imread(img)
    #     elif type(img) == np.ndarray:
    #         frame1 = img.copy()
    #     else:
    #         print('Please input image or image path.')
    #     # frame1=frame.copy()
    #     frame = self.IF.homoFiltering(frame1)
    #     frame_expanded = np.expand_dims(frame, axis=0)
    #     (boxes, scores, classes, num) = self.sess.run(
    #         [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
    #         feed_dict={self.image_tensor: frame_expanded})
    #     objs = []
    #     for k in range(int(scores.size)):
    #         obj = []
    #         if scores[0][k]>threshold:
    #             xmin=int(round(boxes[0][k,1]*frame.shape[1]))
    #             ymin=int(round(boxes[0][k,0]*frame.shape[0]))
    #             xmax=int(round(boxes[0][k,3]*frame.shape[1]))
    #             ymax=int(round(boxes[0][k,2]*frame.shape[0]))
    #             obj.append(self.category_index.get(classes[0][k]).get('name').lower())
    #             obj.append(np.array([xmin,ymin,xmax,ymax]))
    #             objs.append(obj)
    #     return frame, objs

    def dataObj(self, img, threshold=0.9):
        if type(img) == str:
            frame = cv2.imread(img)
        elif type(img) == np.ndarray:
            frame = img.copy()
        else:
            print('Please input image or image path.')
        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        objs=[]
        for k in range(int(scores.size)):
            obj = []
            if scores[0][k]>threshold:
                xmin=int(round(boxes[0][k,1]*frame.shape[1]))
                ymin=int(round(boxes[0][k,0]*frame.shape[0]))
                xmax=int(round(boxes[0][k,3]*frame.shape[1]))
                ymax=int(round(boxes[0][k,2]*frame.shape[0]))
                obj.append(self.category_index.get(classes[0][k]).get('name').lower())
                obj.append('Unspecified')
                if ((xmin<=1) or (xmax>=frame.shape[1]-1) or (ymin<=1) or (ymax>=frame.shape[0]-1)):
                    obj.append(1)
                else:
                    obj.append(0)
                obj.append(0)
                obj.append(np.array([xmin,ymin,xmax,ymax]))
                objs.append(obj)
        return objs

class YOLOV4():

    def __init__(self, ckpt_file_path, PRED_NAMES):
        saver = tf.train.import_meta_graph(ckpt_file_path + '.meta', clear_devices=True)
        graph = tf.get_default_graph()
        self.sess = tf.Session()
        # with tf.Session() as sess:
        saver.restore(self.sess, ckpt_file_path)
        self.names, CLASS_NUM = yolo_label.load_names(PRED_NAMES)
        self.inputs = graph.get_tensor_by_name('inputs:0')
        self.pred_boxes = graph.get_tensor_by_name('pred_boxes:0')
        self.pred_scores = graph.get_tensor_by_name('pred_scores:0')
        self.pred_labels = graph.get_tensor_by_name('pred_labels:0')
        self.colors = []
        for i in range(len(self.names)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.colors.append((b, g, r))

    def catchObject_J(self, img, min_score_thresh, visualize_boxes_IO="OFF"):
        if type(img) == str:  # 給自串路徑
            img = cv2.imread(img)
            image = img.copy()
            img1 = cv2.resize(img, (608, 608))
            frame = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
        elif type(img) == np.ndarray:  # 已經是圖片
            image = img.copy()
            img1 = cv2.resize(img.copy(), (608, 608))
            frame = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
        else:
            print('錯誤，Please input image or image path.')
        # Perform the actual detection by running the model with the image as input
        boxes, scores, labels = self.sess.run([self.pred_boxes, self.pred_scores, self.pred_labels], feed_dict={self.inputs: [frame]})
        objs = []
        for k in range(len(boxes)):
            obj = []  # 每次清空
            if scores[k] >= min_score_thresh:
                xmin = int(round(boxes[k][0] * img.shape[1]))
                ymin = int(round(boxes[k][1] * img.shape[0]))
                xmax = int(round(boxes[k][2] * img.shape[1]))
                ymax = int(round(boxes[k][3] * img.shape[0]))
                obj.append(self.names[labels[k]].lower()) # label name
                obj.append(np.array([xmin, ymin, xmax, ymax])) # 座標
                obj.append(np.array([xmax - xmin, ymax - ymin])) # w, h
                obj.append(scores[k]) # 信心值
                objs.append(obj)
        if visualize_boxes_IO == 'ON':  # 判斷FastRCNN內建可試化視窗開啟或關閉
            for i, obj in enumerate(objs):
                xmin, ymin, xmax, ymax = obj[1]
                for ID, classname in self.names.items():
                    if obj[0] == classname:
                        id = ID
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), self.colors[id], 2)
                # cv2.putText(image, obj[0] + '{:2d}%'.format(int(obj[3] * 100)),
                #             (xmin, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors[id], 2, cv2.LINE_AA)
        return image, objs
