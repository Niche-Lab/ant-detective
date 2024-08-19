import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image


class Video:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        # parameters
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps
        self.summary()

    def summary(self):
        print("fps: ", self.fps)
        print("frame_count: ", self.frame_count)
        print("width: ", self.width)
        print("height: ", self.height)
        print("duration: ", self.duration)

    def __del__(self):
        self.cap.release()

    def get_frame(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame

    def get_frame_at_time(self, time):
        frame_number = int(time * self.fps)
        return self.get_frame(frame_number)

    def get_frame_at_percent(self, percent):
        frame_number = int(self.frame_count * percent)
        return self.get_frame(frame_number)

    def show_frame(self, frame_number):
        frame = self.get_frame(frame_number)
        self.imshow(frame)

    def show_frame_at_time(self, time):
        frame = self.get_frame_at_time(time)
        self.imshow(frame)

    def imshow(self, frame, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        plt.imshow(frame[:, :, [2, 1, 0]])


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DIR_IN = os.path.join(ROOT, "raw_gopro")
DIR_OUT = os.path.join(ROOT, "dod_prelim")
os.chdir(DIR_IN)
os.listdir()

# pre_a_03.mov

## frame1
mov = Video("pre_a_03.mov")
frame = mov.get_frame(5 * mov.fps + 20)
frame = frame[50:700, 100:800, [2, 1, 0]]
Image.fromarray(frame).save(os.path.join(DIR_OUT, "frame1.jpg"))

## frame2
mov = Video("pre_a_02.mov")
frame = mov.get_frame(19 * mov.fps + 20)
frame = frame[200:, 170:820, [2, 1, 0]]
plt.figure(figsize=(10, 10))
plt.imshow(frame)
Image.fromarray(frame).save(os.path.join(DIR_OUT, "frame2.jpg"))


# GX013734.mov
mov = Video("GX013734.mov")
frame = mov.get_frame(40 * mov.fps)
frame = frame[:, :, [2, 1, 0]]
Image.fromarray(frame).save(os.path.join(DIR_OUT, "gx013734.jpg"))
