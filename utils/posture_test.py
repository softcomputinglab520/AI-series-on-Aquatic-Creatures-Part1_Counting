"""
測試單一frame
"""
from utils.Detector import *
import cv2
import random
import numpy as np
import math
import utils.posture_config as pos_con
import utils.posture_utils as pos_util

visualize_boxes_IO = 'ON'#FastRCNN內建可試化視窗開啟或關閉，ON or OFF
# MODEL_PATH = './Faster_RCNN_Model/Ver01/'
# detector = Faster_RCNN(MODEL_PATH + 'frozen_inference_graph.pb', MODEL_PATH + 'labelmap.pbtxt')
MODEL_NAME = 'tilapia'
detector = YOLOV4('./YOLOV4_Model/Ver01/' + MODEL_NAME + '_models/model', './YOLOV4_Model/Ver01/' + MODEL_NAME + '.names')
names, CLASS_NUM = yolo_label.load_names('./YOLOV4_Model/Ver01/' + MODEL_NAME + '.names')
name_list = [names[x] for x in names]
VIDEO_PATH = './videos/'
VIDEO_NAME = '110.mp4'
PATH_TO_VIDEO = VIDEO_PATH + VIDEO_NAME
video = cv2.VideoCapture(PATH_TO_VIDEO)
min_score_thresh = 0.7
a_t = []
frame_count = 0
log_list = []
count = 0
Abnormal_behavior_count = 0
cc = 0
cccount = 0
cvshowtext = []
count_abnormal = 0
width1 = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open video file
# Start_frame = 554
Start_frame = 554
video.set(cv2.CAP_PROP_POS_FRAMES, Start_frame) #設定影片起始位置
_, frame1 = video.read()
height = height1
width = int(width1 / 2)
frame = frame1[:, int(width1 / 2):]
image, objs = detector.catchObject_J(frame, min_score_thresh, visualize_boxes_IO)  # 物件偵測器
frame_test = frame.copy()
main_box_name = 'tilapia'

def getEachLabelBndData(name_list, obj_list):
    total_data = {}
    for label in name_list:
        total_data[label] = []
    for obj in obj_list:
        for label in name_list:
            if obj[0] == label:
                total_data[label].append(obj[1])
    return total_data

all_bnd = getEachLabelBndData(name_list, objs)
def part_match(parts_list, body):
    tilapia_xmin, tilapia_ymin, tilapia_xmax, tilapia_ymax = body
    final_xy = []
    for j in range(len(parts_list)):
        xmin, ymin, xmax, ymax = parts_list[j]
        if (tilapia_xmax > int((xmax + xmin) / 2) > tilapia_xmin) and (tilapia_ymax > int((ymax + ymin) / 2) > tilapia_ymin) and float(((xmax-xmin)*(ymax-ymin))/((tilapia_xmax-tilapia_xmin)*(tilapia_ymax-tilapia_ymin))) > 0.25
            final_xy.append(parts_list[j])
    return final_xy

def getCompleteFish(all_bnd, main_box_name):
    fish = []
    for body in all_bnd[main_box_name]:
        fish.append({
            main_box_name: [body],
            'analfin': part_match(all_bnd['analfin'], body),
            'dorsalfin': part_match(all_bnd['dorsalfin'], body),
            'fin': part_match(all_bnd['fin'], body),
            'head': part_match(all_bnd['head'], body),
            'eye': part_match(all_bnd['eye'], body),
            'mouth': part_match(all_bnd['mouth'], body),
            'tail': part_match(all_bnd['tail'], body),
        })
    return fish
fish_list = getCompleteFish(all_bnd, main_box_name)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(fish_list))]
for j, f in enumerate(fish_list):
    for part in f:
        part_xy = f[part]
        if len(part_xy) != 0:
            xmin, ymin, xmax, ymax = part_xy[0]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colors[j], 2)

# fish = fish_list[0]


def cart2Polar(center, x):
    r = math.sqrt(math.pow(x[0] - center[0], 2) + math.pow(x[1] - center[1], 2))
    theta = math.atan2(x[1] - center[1], x[0] - center[0]) / math.pi * 180  # 轉換爲角度
    return r, theta


def quantizeDegrees(theta):
    if 22.5 <= theta < 67.5:
        return 45
    elif 67.5 <= theta < 112.5:
        return 90
    elif 112.5 <= theta < 157.5:
        return 135
    elif 157.5 <= theta < 202.5:
        return 180
    elif 202.5 <= theta < 247.5:
        return 225
    elif 247.5 <= theta < 292.5:
        return 270
    elif 292.5 <= theta < 337.5:
        return 315
    else:
        return 0


def getHeadDegrees(part_degrees, fish):
    if part_degrees['head'] is None:
        if part_degrees['eye'] is None:
            if part_degrees['mouth'] is None:
                if part_degrees['tail'] is None:
                    af_c = np.array([int((fish['analfin'][0][0] + fish['analfin'][0][2]) / 2),
                                     int((fish['analfin'][0][1] + fish['analfin'][0][3]) / 2)])
                    f_c = np.array([int((fish['fin'][0][0] + fish['fin'][0][2]) / 2),
                                    int((fish['fin'][0][1] + fish['fin'][0][3]) / 2)])
                    _, h_theta = cart2Polar(af_c, f_c)
                else:
                    h_theta = part_degrees['tail'] + 180
            else:
                h_theta = part_degrees['mouth']
        else:
            h_theta = part_degrees['eye']
    else:
        h_theta = part_degrees['head']
    if h_theta >= 360:
        h_theta -= 360
    h_theta = -h_theta # 影像直角座標與正常直角座標的Y軸方向相差180度，故將角度加上負號調整至正常直角座標
    if h_theta <= 0:
        h_theta += 360
    return quantizeDegrees(h_theta)


def checkPhaseSequence(pos):
    if pos in pos_con.positive_phase_sequence:
        return 1
    elif pos in pos_con.negative_phase_sequence:
        return -1
    else:
        return 0


def getPartReference(degrees_list, part_degrees):
    degrees_list.sort()
    min_val1, min_val2, min_val3 = degrees_list[0], degrees_list[1], degrees_list[2]
    first_sym = ''
    second_sym = ''
    third_sym = ''
    for part in part_degrees:
        if min_val1 == part_degrees[part]:
            first_sym = pos_con.symbol_table[part]
        if min_val2 == part_degrees[part]:
            second_sym = pos_con.symbol_table[part]
        if min_val3 == part_degrees[part]:
            third_sym = pos_con.symbol_table[part]
    if (first_sym == 't' and second_sym == 'h') or (first_sym == 'h' and second_sym == 't') \
            or (first_sym == 'af' and second_sym == 'df') or (first_sym == 'f' and second_sym == 'df')\
            or (first_sym == 'df' and second_sym == 'af') or (first_sym == 'df' and second_sym == 'f'):
        if abs(min_val2 - min_val3) <= 180:
            part_ref = second_sym + '>' + third_sym
        else:
            part_ref = third_sym + '>' + second_sym
    else:
        if abs(min_val1 - min_val2) <= 180:
            part_ref = first_sym + '>' + second_sym
        else:
            part_ref = second_sym + '>' + first_sym
    return part_ref


def getPartDegrees(fish, main_box_name):
    degrees_list = []
    part_degrees = {}
    body_c = np.array([int((fish[main_box_name][0][0] + fish[main_box_name][0][2]) / 2),
                       int((fish[main_box_name][0][1] + fish[main_box_name][0][3]) / 2)])
    for part in fish:
        if part != main_box_name and part != 'posture_vector':
            if len(fish[part]) != 0:
                part_c = np.array(
                    [int((fish[part][0][0] + fish[part][0][2]) / 2), int((fish[part][0][1] + fish[part][0][3]) / 2)])
                _, theta = cart2Polar(body_c, part_c)
                if theta < 0:
                    theta += 360
                if ((part == 'eye') and (part_degrees['head'] != None)):
                    degrees_list.append(4000)
                    part_degrees[part] = None
                elif ((part == 'mouth') and (part_degrees['eye'] != None)):
                    degrees_list.append(4000)
                    part_degrees[part] = None
                else:
                    degrees_list.append(theta)
                    part_degrees[part] = theta
            else:
                degrees_list.append(4000)
                part_degrees[part] = None
    return degrees_list, part_degrees


def decidePostureVector(fish, main_box_name):
    degrees_list, part_degrees = getPartDegrees(fish, main_box_name)
    part_ref = getPartReference(degrees_list, part_degrees)
    gamma = checkPhaseSequence(part_ref)
    if gamma != 0:
        h_theta = getHeadDegrees(part_degrees, fish)
        x = math.cos(h_theta / 180 * math.pi)
        y = math.sin(h_theta / 180 * math.pi)
        fish['posture_vector'] = [round(x, 3), round(y, 3), gamma]
    else:
        fish['posture_vector'] = [0, 0, gamma]

for fish in fish_list:
    pos_util.decidePostureVector(fish, main_box_name)

for j, f in enumerate(fish_list):
    for part in f:
        part_xy = f[main_box_name]
        xmin, ymin, xmax, ymax = part_xy[0]
        cv2.rectangle(frame_test, (xmin, ymin), (xmax, ymax), colors[j], 2)
        cv2.putText(frame_test, 'PosVec:' + str(f['posture_vector']), (xmin, ymin + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[j], 1, cv2.LINE_AA)

cv2.imshow('Object detector', image)
cv2.imshow('frame', frame)
cv2.imshow('frame_test', frame_test)
# temp = np.hstack((image, frame, frame_test))
# cv2.imwrite('./' + VIDEO_NAME + '_Part_Pos.jpg', temp)
cv2.waitKey(0)
# Clean up
video.release()
cv2.destroyAllWindows()
