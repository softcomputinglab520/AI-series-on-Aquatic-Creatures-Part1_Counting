import cv2
import os
import numpy as np


def filename(i, num):
    num1 = str(num)
    amount = len(num1)
    temp = ''
    for t in range(amount - len(str(i))):
        temp += '0'
    temp += str(i)
    return temp


def video2img(video, second, video_name, out, isDouble=False, writeDouble=False):
    if not os.path.exists(out):
        os.makedirs(out)
    frames = video.get(7)
    fps = video.get(5)
    index = list(np.arange(1, frames, fps * second))
    # for i, id in enumerate(index):
    for i in range(len(index)):
        video.set(cv2.CAP_PROP_POS_FRAMES, index[i] + 1)
        # video.set(cv2.CAP_PROP_POS_FRAMES, id + 1)
        _, frame1 = video.read()
        if isDouble:
            size = frame1.shape
            w = size[1]
            if writeDouble:
                frame_L = frame1[:, :int(w / 2)]
                frame_R = frame1[:, int(w / 2):]
                cv2.imwrite(out + video_name[:len(video_name) - 3] + filename(i, frames) + '_L.jpg', frame_L)
                cv2.imwrite(out + video_name[:len(video_name) - 3] + filename(i, frames) + '_R.jpg', frame_R)
            else:
                frame = frame1[:, :int(w / 2)]
                cv2.imwrite(out + video_name[:len(video_name) - 3] + filename(i, frames) + '.jpg', frame)
        else:
            cv2.imwrite(out + video_name[:len(video_name) - 3] + filename(i, frames) + '.jpg', frame1)


def videos2img(path, out, second=1, isDouble=False, writeDouble=False):
    if not os.path.exists(out):
        os.makedirs(out)
    file_names = [x for x in os.listdir(path)]
    for video_name in file_names:
        # print(video_name)
        video = cv2.VideoCapture(path + video_name)
        video2img(video, second, video_name, out, isDouble, writeDouble)
        video.release()

