import pandas as pd
import matplotlib.pyplot as plt
import cv2
from highlight_table import highlight_table_on_frame
import sys
import numpy as np
import os


# python3 demo.py images_for_dataset/layout.csv

if __name__ == '__main__':

    df = pd.read_csv(sys.argv[1], header=None)
    n = len(df)

    for i in range(n):
        path = df.iloc[i][0]
        hull = []
        for j in range(4):
            hull.append([df.iloc[i][2 * j + 1], df.iloc[i][2 * j + 2]])
        hull = np.array(hull)
        img_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[1]), path))

        frame = cv2.imread(img_path)
        highlight_table_on_frame(frame, hull)

        plt.imshow(frame[:, :, ::-1])
        plt.show()
