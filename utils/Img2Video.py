import cv2
import os
import numpy as np

class Img2Video():

    def __init__(self, imgs, writefilename):
        self.imgs = imgs
        self.writefilename = writefilename

    # def combine(self, path1, path2):
    #     temp = []
    #     imgpath1 = os.listdir(path1)
    #     imgpath2 = os.listdir(path2)
    #     for i in range(len(imgpath1)):
    #         img1 = cv2.resize(cv2.imread(path1 + '/' + imgpath1[i]), (960, 480))
    #         img2 = cv2.resize(cv2.imread(path2 + '/' + imgpath2[i]), (960, 480))
    #         img = np.hstack((img1, img2))
    #         temp.append(img)
    #     return temp

    def make_video(self, images, writepath, outimg=None, fps=10, size=None,
                   is_color=True, format="X264"):
        """
        Create a video from a list of images.

        @param      outvid      output video
        @param      images      list of images to use in the video
        @param      fps         frame per second
        @param      size        size of each frame
        @param      is_color    color
        @param      format      see http://www.fourcc.org/codecs.php
        @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

        The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
        By default, the video will have the size of the first image.
        It will resize every image to this size before adding them to the video.
        """
        from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
        fourcc = VideoWriter_fourcc(*format)
        vid = None
        for image in images:
            img = image
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(writepath + ".mp4", fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        if vid is not None:
            vid.release()
        return vid

    def excute(self, fps=20):
        # imgs = self.combine(self.root + self.folder1, self.root + self.folder2)
        self.make_video(self.imgs, self.writefilename, fps=fps)
