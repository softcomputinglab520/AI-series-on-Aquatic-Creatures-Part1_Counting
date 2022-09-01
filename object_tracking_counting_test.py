import cv2
from utils.Detector import Faster_RCNN
from utils import trackfunction as tracker
from yolov4API.YingRen_Yolov4API import Yolov4YingRen
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('--detector', type=str, default="yolov4")
    parser.add_argument('--modelname', type=str, default="lobster")
    parser.add_argument('--video', type=str, default="./videos/002.mp4", help="video URL")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--lost_target', type=int, default=5, help="The maximum number of lost frames to track the target")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    if args.detector == "fasterrcnn":
        detector = Faster_RCNN(f"faster_rcnn/{args.modelname}.pb", f"faster_rcnn/{args.modelname}.pbtxt")
    elif args.detector == "yolov4":
        detector = Yolov4YingRen(f"yolov4/{args.modelname}.cfg", f"yolov4/{args.modelname}.data",
                                 f"yolov4/{args.modelname}.weights")
    else:
        pass
    targets = []
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    if args.output is not None:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        format = "X264"
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*format)
        is_color = True
        size = int(video.get(3)), int(video.get(4))
        vid = cv2.VideoWriter(f"{args.output}/rs.mp4", fourcc, float(fps), size, is_color)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        current_frame = int(video.get(1))
        objs = detector.catchObject(frame, .99)
        if len(objs) == 0:
            continue
        if len(targets) == 0:
            tracker.addTarget(targets, objs, current_frame)
        else:
            update_set, add_set, predict_index_set = tracker.targetMatching(targets, objs)
            tracker.targetUpdate(targets, update_set)
            tracker.targetPredict(targets, predict_index_set, args.lost_target)
            tracker.addTarget(targets, add_set, current_frame)
        now_count = 0
        for t, target in enumerate(targets):
            if target["isTarcking"]:
                if target["state"] == 'grazing' and not "isGrazingCount" in target:
                    target["isGrazingCount"] = True
                if target["state"] == 'turn' and not "isTurnCount" in target:
                    target["isTurnCount"] = True
                if len(target["position"]) > 10:
                    now_count += 1
                    if not "isCount" in target:
                        count += 1
                        target["isCount"] = True
                    tracker.drawBox(frame, target["position"][-1], t)
                    tracker.drawTrace(frame, target)
        cv2.putText(frame, f"history counted : {count}", (20, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"current count : {now_count}", (20, 50), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if args.output is not None:
            vid.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cv2.destroyWindow("frame")
    video.release()
    if args.output is not None:
        vid.release()


if __name__ == '__main__':
    main()
