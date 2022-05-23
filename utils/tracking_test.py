from utils.Detector import *
import utils.posture_utils as pos_util
import utils.tracker_J as mytracker
import cv2
import time

visualize_boxes_IO = 'ON'#FastRCNN內建可試化視窗開啟或關閉，ON or OFF
t_amount = 30#取追蹤的數量，如為0代表關閉追蹤功能
# MODEL_PATH = './Faster_RCNN_Model/Ver01/'
# detector = Faster_RCNN(MODEL_PATH + 'frozen_inference_graph.pb', MODEL_PATH + 'labelmap.pbtxt')
MODEL_NAME = 'tilapia'
detector = YOLOV4('./YOLOV4_Model/Ver01/' + MODEL_NAME + '_models/model', './YOLOV4_Model/Ver01/' + MODEL_NAME + '.names')
names, CLASS_NUM = yolo_label.load_names('./YOLOV4_Model/Ver01/' + MODEL_NAME + '.names')
name_list = [names[x] for x in names]
VIDEO_PATH = './videos/'
VIDEO_NAME = '110'
PATH_TO_VIDEO = VIDEO_PATH + VIDEO_NAME + '.mp4'
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
main_box_name = 'tilapia'
N_t_all_min = 0
distance_min = 0
tracking_start = 0
targets_temp = []
angle = 0
lost_target = 5#追蹤目標失去多少frame結束追蹤
targets_history = []
# Open video file

while (video.isOpened()):
    start = time.time()
    ret, frame1 = video.read()
    if not ret:
        break
    post_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    height = height1
    width = int(width1 / 2)
    frame = frame1[:, int(width1 / 2):]
    frame_test = frame.copy()
    image, objs = detector.catchObject_J(frame, min_score_thresh, visualize_boxes_IO)  # 物件偵測器
    all_bnd = pos_util.getEachLabelBndData(name_list, objs)
    fish_list = pos_util.getCompleteFish(all_bnd)
    for fish in fish_list:
        pos_util.decidePostureVector(fish, main_box_name)
    N, N1 = mytracker.targetUpdate01(fish_list, targets_temp, t_amount, post_frame, main_box_name)
    for i, target in enumerate(targets_temp):  # list都會相通長度
        trajectory, all_save, class_name, fish_data = target['trajectory'], target['all_save'], target['classname'], target['fish_data']
        if all_save[-1][0] != post_frame:
            target_tracking, target_color, distance_min, C_C, class_name_temp, fish_data_temp = mytracker.targetMatch01(N, N1, trajectory, all_save,
                                                                                            class_name, fish_list, fish_data)
            trajectory.append(np.array(target_tracking))
            all_save.append([post_frame, C_C, round(distance_min * 10) / 10, round(distance_min * 10) / 10,
                             round(angle * 180 / np.pi * 10) / 10])
            class_name.append(class_name_temp)
            fish_data.append(fish_data_temp)
            # Speed_list_t.append(int(round(distance_min)))
            # 累積大於5個frame未能從RCNN得到候選框，動量預測的追蹤框存在負數，候選框跟目標追蹤框中心的距離大於追蹤框的長或寬的一半
            if C_C >= lost_target or any(np.array(target_tracking) < 0) or distance_min > min(target_tracking[2] / 2,
                                                                                              target_tracking[3] / 2):
                # 結束追蹤
                target_temp = targets_temp[i].copy()
                targets_history.append(target_temp)
                targets_temp.pop(i)
            elif C_C < lost_target and post_frame - tracking_start > 5:  # 負責顯示追蹤結果
                c, r, w, h = target_tracking
                # 框出魚跟寫上Fish
                cv2.rectangle(frame_test, (c, r), (c + w, r + h), (0, 255, 255), 5)
                for i in range(1, len(trajectory)):  # 魚的軌跡
                    cv2.line(frame_test, (int(trajectory[i - 1][0] + trajectory[i - 1][2] / 2),
                                      int(trajectory[i - 1][1] + trajectory[i - 1][3] / 2)), (
                             int(trajectory[i][0] + trajectory[i][2] / 2),
                             int(trajectory[i][1] + trajectory[i][3] / 2)), (0, 255, 0), 2)
    end = time.time()
    fps = 1. / (end - start)
    cv2.putText(frame_test, 'FPS: %.2f' % fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, '#' + str(int(video.get(1))), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Object detector', image)
    cv2.imshow('frame', frame)
    cv2.imshow('frame_test', frame_test)
    temp = np.hstack((image, frame, frame_test))
    if cv2.waitKey(1) & 0xff == 27:
        break
# Clean up
video.release()
cv2.destroyAllWindows()
