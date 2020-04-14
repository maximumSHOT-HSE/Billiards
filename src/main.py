from hmmlearn import hmm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from copy import deepcopy

from src.video_operations import save_frames_as_video

np.random.seed(42)


if __name__ == '__main__':
    with open('layout.txt', 'r') as fin:
        capture = cv2.VideoCapture('resources/001/raw/video001_0.mp4')

        for line in fin:
            y1, x1, y2, x2, y3, x3, y4, x4 = [int(x) for x in line.strip().split(' ')]
            response, frame = capture.read()

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), 4)

            cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), 4)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), 4)

            cv2.line(frame, (x3, y3), (x4, y4), (0, 255, 0), 4)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), 4)

            cv2.line(frame, (x4, y4), (x1, y1), (0, 255, 0), 4)
            cv2.circle(frame, (x4, y4), 10, (0, 0, 255), 4)

            plt.imshow(frame[:, :, ::-1])
            plt.show()
            exit(0)
