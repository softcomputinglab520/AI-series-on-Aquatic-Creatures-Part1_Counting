import cv2
from utils.Detector import Faster_RCNN
from utils import trackfunction as tracker
from yolov4API.YingRen_Yolov4API import Yolov4YingRen
import argparse


def main():
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--detector', type=str, default="yolov4")
    parser.add_argument('--modelname', type=str, default="lobster")
    parser.add_argument('--video', type=str, default="./videos/002.mp4", help='video URL')
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    if args.detector == "fasterrcnn":
        detector = Faster_RCNN(f"faster_rcnn/{args.modelname}.pb", f"faster_rcnn/{args.modelname}.pbtxt")
    elif args.detector == "yolov4":
        detector = Yolov4YingRen(f"yolov4/{args.modelname}.cfg", f"yolov4/{args.modelname}.data", f"yolov4/{args.modelname}.weights")
    else:
        pass

    while 1:
        ret, frame = video.read()
        objs = detector.catchObject(frame, .99)
        tracker.drawBoxes(frame, objs)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cv2.destroyWindow("frame")
    video.release()


if __name__ == '__main__':
    main()
