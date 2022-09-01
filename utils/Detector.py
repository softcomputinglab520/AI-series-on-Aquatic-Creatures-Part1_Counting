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
        """
        Faster-RCNN class init setting
        :param CKPTPATH: the model check point path or frozen model pb file path
        :param LABELPATH: the model label map path
        """
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

    def catchObject(self, img, threshold=0.9):
        """
        The function of catch objects via threshold, which was the minimized confidence score proposed by model.
        :param img: the image from video or webcam frame
        :param threshold: the minimized confidence score proposed by model
        :return: the objects (data type list) every object contained
        [
        label name (data type string),
        bounding box position (data type numpy array, np.array([xmin, ymin, xmax, ymax]))
        ]
        """
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


class YOLOV4():

    def __init__(self, ckpt_file_path, PRED_NAMES):
        """
        YOLO v4 class init setting
        :param ckpt_file_path: the model check point path or frozen model pb file path
        :param PRED_NAMES: the model label text path
        """
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
            if scores[k] >= threshold:
                xmin = int(round(boxes[k][0] * img.shape[1]))
                ymin = int(round(boxes[k][1] * img.shape[0]))
                xmax = int(round(boxes[k][2] * img.shape[1]))
                ymax = int(round(boxes[k][3] * img.shape[0]))
                obj.append(self.names[labels[k]].lower())
                obj.append(np.array([xmin, ymin, xmax, ymax]))
                objs.append(obj)
        return objs


class YOLOV4_tiny():

    def __init__(self, CKPTPATH, LABELPATH):
        """
        YOLO v4 tiny class init setting
        :param CKPTPATH: the model check point path or frozen model pb file path
        :param LABELPATH: the model label text path
        """
        PATH_TO_CKPT = CKPTPATH
        PATH_TO_LABELS = LABELPATH
        self.names, CLASS_NUM = yolo_label.load_names(PATH_TO_LABELS)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                self.sess = tf.Session(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name('inputs:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('output_boxes:0')
        self.colors = []
        for i in range(CLASS_NUM):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
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
        if type(img) == str:  # 給自串路徑
            img = cv2.imread(img)
            image = img.copy()
            frame = cv2.resize(img, (416, 416))
        elif type(img) == np.ndarray:  # 已經是圖片
            image = img.copy()
            frame = cv2.resize(img.copy(), (416, 416))
        else:
            print('錯誤，Please input image or image path.')
        # Perform the actual detection by running the model with the image as input
        frame_expanded = np.expand_dims(frame, axis=0)
        output = self.sess.run(self.detection_boxes, feed_dict={self.image_tensor: frame_expanded})
        boxes, scores, labels = self.outputConvert(output, threshold)
        h, w, _ = image.shape
        h_ratio = h / 416
        w_ratio = w / 416
        objs = []
        for k, box in enumerate(boxes):
            obj = []
            xmin = int(round(box[0] * w_ratio))
            ymin = int(round(box[1] * h_ratio))
            xmax = int(round(box[2] * w_ratio))
            ymax = int(round(box[3] * h_ratio))
            obj.append(self.names[labels[k]].lower())
            obj.append(np.array([xmin, ymin, xmax, ymax]))
            objs.append(obj)
        return image, objs

    def outputConvert(output, threshold):
        boxes = []
        scores = []
        labels = []
        for out in output[0]:
            if out[4] >= threshold:
                boxes.append([out[0], out[1], out[2], out[3]])
                scores.append(out[4])
                temp = list(out[5:])
                labels.append(temp.index(max(temp)))
        return boxes, scores, labels
