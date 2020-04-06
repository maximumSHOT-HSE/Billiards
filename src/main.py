from hmmlearn import hmm
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(42)

if __name__ == '__main__':
    cap = cv2.VideoCapture('resources/001/video001.mp4')

    s = 0
    c = 0

    while cap.isOpened():
        response, frame = cap.read()
        if not response:
            break
        s += np.mean(frame)
        c += 1

    print(s)
    print(c)
