
from utils.Detector import *
import cv2
import time 

visualize_boxes_IO = 'ON'#FastRCNN內建可試化視窗開啟或關閉，ON or OFF
MODEL_PATH = './model/Ver01/'
detector = Faster_RCNN(MODEL_PATH + 'frozen_inference_graph.pb', MODEL_PATH + 'labelmap.pbtxt')
VIDEO_PATH = './video/'
VIDEO_NAME = '001.mp4'
PATH_TO_VIDEO = VIDEO_PATH + VIDEO_NAME
video = cv2.VideoCapture(PATH_TO_VIDEO)
min_score_thresh = 0.99
width1 = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open video file
while (video.isOpened()):
    ret, frame1 = video.read()
    if not ret:
        break
    start = time.time()
    height = height1
    width = int(width1)
    frame = frame1
    image, objs = detector.catchObject_J(frame, min_score_thresh, visualize_boxes_IO)  # 物件偵測器
    
    for i, obj in enumerate(objs):
        xmin, ymin, xmax, ymax = obj[1][0]
        if obj[0] == 'lobster':
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(image, obj[0], (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 1, cv2.LINE_AA)
    
    end = time.time()
    seconds = end - start
    fps = 1 / seconds
    cv2.putText(image, "FPS: {0}".format(round(fps,1)), (0, 15), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 0), 1, cv2.LINE_AA)
    print(fps)
    
    cv2.imshow('Object detector', image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xff == 27:
        break

# Clean up
video.release()
cv2.destroyAllWindows()
