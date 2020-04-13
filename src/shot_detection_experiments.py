from hmmlearn import hmm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

from copy import deepcopy
from collections import deque

from src.video_operations import save_frames_as_video
from scipy.spatial import ConvexHull

np.random.seed(42)


# assume that table has blue color and table cover the biggest blue area
# cell is blue iff (B - R > threshold) & (B - G > threshold)
def blue_table_mask(frame: np.ndarray, threshold: int = 10):
    R = frame[:, :, 0].astype(int)
    G = frame[:, :, 1].astype(int)
    B = frame[:, :, 2].astype(int)
    return (B - R > threshold) & (B - G > threshold)


def green_table_mask(frame: np.ndarray, threshold: int = 25):
    R = frame[:, :, 0].astype(int)
    G = frame[:, :, 1].astype(int)
    B = frame[:, :, 2].astype(int)
    return (G - R > threshold) & (G - B > threshold)


# takes video, masks blue parts, saves generated two-colored video
def exp1(input_video_path: str, output_video_path: str, verbose=True):
    capture = cv2.VideoCapture(input_video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    masked_frames = []

    while capture.isOpened():
        response, frame = capture.read()
        if not response:
            break
        frame = frame[:, :, ::-1]  # RGB
        mask = blue_table_mask(frame).astype(dtype='uint8')

        masked_frame = frame
        masked_frame[mask == True] = [255, 255, 255]
        masked_frame[mask == False] = [0, 0, 0]

        masked_frames.append(masked_frame)

        if verbose:
            print(len(masked_frames))

    save_frames_as_video(output_video_path, masked_frames, fps)


# takes video, applies blue table mask, applies Canny, applies Hough transform, draws Hugh lines
def exp2(input_video_path: str, output_video_path: str, verbose=True):
    capture = cv2.VideoCapture(input_video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    imgs = []

    while capture.isOpened():
        response, frame = capture.read()

        if not response:
            break

        frame = frame[:, :, ::-1]

        mask = blue_table_mask(frame).astype('int')

        masked_frame = deepcopy(frame)
        masked_frame[mask == True] = [255, 255, 255]
        masked_frame[mask == False] = [0, 0, 0]

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        min_line_length = 200
        max_line_gap = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, min_line_length, max_line_gap)

        img = np.zeros_like(frame)

        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        imgs.append(img)

        if verbose:
            print(len(imgs))

    save_frames_as_video(output_video_path, imgs, fps)


# takes 2d array of bools, finds connected component of Trues with the largest area and leaves only that component
# so other component will be False
# PS. Two pixels are connected iff they have commond side
def find_largest_area_component_mask(mask, center_ratio_size: float = 0.5):
    q = deque()
    n, m = mask.shape

    skip_pxl_n = int(n * (1 - center_ratio_size) / 2)
    skip_pxl_m = int(m * (1 - center_ratio_size) / 2)

    color = np.zeros_like(mask)
    current_color = 0
    best_component_size, best_component_color = 0, 0
    for si in range(skip_pxl_n, n - skip_pxl_n):
        for sj in range(skip_pxl_m, m - skip_pxl_m):
            if not mask[si, sj] or color[si, sj] != 0:
                continue
            current_color += 1
            component_size = 0
            q.clear()
            q.append((si, sj))

            while len(q) > 0:
                vi, vj = q.pop()
                if vi < 0 or vi >= n or vj < 0 or vj >= m or color[vi, vj] != 0 or not mask[vi, vj]:
                    continue
                color[vi, vj] = current_color
                component_size += 1
                q.append((vi - 1, vj))
                q.append((vi + 1, vj))
                q.append((vi, vj - 1))
                q.append((vi, vj + 1))

            if component_size > best_component_size:
                best_component_size, best_component_color = component_size, current_color
    return color == best_component_color


def find_table_mask(
        frame,
        same_color_threshold: int = 4000,
        table_color_center_ratio_size: float = 0.1
):
    n, m, _ = frame.shape

    skip_pxl_n = int(n * (1 - table_color_center_ratio_size) / 2)
    skip_pxl_m = int(m * (1 - table_color_center_ratio_size) / 2)

    table_color = np.mean(frame[skip_pxl_n: n - skip_pxl_n, skip_pxl_m: m - skip_pxl_m, :], axis=(0, 1)) \
        .astype(dtype=int).clip(0, 255)

    return (np.sum((frame - table_color) ** 2, axis=2) < same_color_threshold).astype(int)


def find_table_polygon(
        frame,
        largest_component_center_ratio_size: int = 0.5
):
    frame = frame[:, :, ::-1]

    mask = find_table_mask(frame)
    mask = find_largest_area_component_mask(mask, largest_component_center_ratio_size)

    masked_frame = deepcopy(frame)
    masked_frame[mask == True] = [255, 255, 255]
    masked_frame[mask == False] = [0, 0, 0]

    # return masked_frame

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_length = 200
    max_line_gap = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, min_line_length, max_line_gap)

    points = np.array([(0, 0) for _ in range(2 * len(lines[:, 0, :]))])

    for i, (x1, y1, x2, y2) in enumerate(lines[:, 0, :]):
        points[2 * i] = (x1, y1)
        points[2 * i + 1] = (x2, y2)

    return points[ConvexHull(points).vertices]


# finds mask for convex polygon on the image of given size
def find_convex_hull_mask(size, hull):
    n, m = size
    mask = np.zeros((n, m))

    i = 0
    I = 0
    # finds lefties and rightest points in the hull
    for q, (y, x) in enumerate(hull):
        if x < hull[i][1]:
            i = q
        if x > hull[I][1]:
            I = q
    j = i
    lx = hull[i][1]
    rx = hull[I][1]

    def get_prev(_id):
        return (_id - 1) if (_id > 0) else (len(hull) - 1)

    def get_next(_id):
        return (_id + 1) % len(hull)

    def get_y_on_segment(p1, p2, x, vertical_segment_policy: str = 'up'):
        y1, x1 = p1
        y2, x2 = p2
        if x1 > x2 or (x1 == x2 and y1 > y2):
            x1, y1, x2, y2, p1, p2 = x2, y2, x1, y1, p2, p1  # swap
        if x1 == x2:
            return y1 if vertical_segment_policy == 'down' else y2
        alpha = (x - x1) / (x2 - x1)
        return int(np.round((1 - alpha) * y1 + alpha * y2))

    for x in range(lx, rx + 1):
        while hull[i][1] < x:
            i = get_prev(i)
        while hull[j][1] < x:
            j = get_next(j)
        yi = get_y_on_segment(hull[i], hull[get_next(i)], x, 'up')
        yj = get_y_on_segment(hull[j], hull[get_prev(j)], x, 'down')
        if yi > yj:
            yi, yj = yj, yi
        mask[x, yi: yj + 1] = 1

    return mask


# color of the table is the mean color among pixels in the some center part of the frame
# table mask is the set of pixels with almost the same color (some metric, e.g. euclid distance)

# takes frame, cuts off 15% of pixels from all sides (left, right, top, bottom),
# finds largest connected component and removes from mask other components
# center_ratio_size is the share of the image which will be cropped before finding
# largest component, finds convex hull
# as the result of experiment initial video with table layout will be saved
def exp3(input_video_path: str, output_video_path):
    capture = cv2.VideoCapture(input_video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    imgs = []

    while capture.isOpened():
        response, frame = capture.read()

        if not response:
            break

        hull = find_table_polygon(deepcopy(frame))
        hull_size = len(hull)

        hull_mask = find_convex_hull_mask(frame.shape[: 2], hull)
        table_mask = find_table_mask(frame)

        table_color = np.mean(frame[(hull_mask * table_mask) == 1], axis=0).astype(int)
        player_mask = find_largest_area_component_mask((1 - hull_mask * table_mask) * hull_mask, center_ratio_size=1)

        img = frame
        img[player_mask == 1] = table_color
        img[hull_mask == 0] = frame[hull_mask == 0]

        for i in range(hull_size):
            x1, y1 = hull[i]
            x2, y2 = hull[(i + 1) % hull_size]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 4)

        plt.figure(figsize=(20, 20))
        plt.imshow(img[:, :, ::-1])
        plt.show()
        break

        imgs.append(img)

    # save_frames_as_video(output_video_path, imgs, fps)


def test_kate_images():
    root = 'resources/images'
    blue_tables = os.path.join(root, 'blue_table')
    green_tables = os.path.join(root, 'green_table')

    # for t in os.listdir(blue_tables):
    #     print(os.path.join(blue_tables, t))
    #     img = cv2.imread(os.path.join(blue_tables, t))
    #     img = find_table_polygon(img, 0.5, 'blue', threshold=30)
    #     # cv2.imwrite(os.path.join(blue_tables, t[:-4] + '_labeled.jpg'), img)
    #     plt.imshow(img)
    #     plt.show()

    for t in os.listdir(green_tables):
        if '6' not in t:
            continue
        print(os.path.join(green_tables, t))
        img = cv2.imread(os.path.join(green_tables, t))
        img = find_table_polygon(img, 0.5, 'green', same_color_threshold=10)
        # cv2.imwrite(os.path.join(green_tables, t[:-4] + '_labeled.jpg'), img)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    for i in range(29):
        if i == 0:
            exp3(f'resources/001/raw/video001_{i}.mp4', f'resources/001/exp3/video001_{i}_exp3.mp4')
    # for i in range(24):
    #     exp3(f'resources/002/raw/video002_{i}.mp4', 'tmp.mp4')
    # for i in range(107):
    #     exp3(f'resources/003/raw/video003_{i}.mp4', 'tmp.mp4')
    # for i in range(5):
    #     exp3(f'resources/004/raw/video004_{i}.mp4', 'tmp.mp4')
    # for i in range(109):
    #     exp3(f'resources/007/raw/video007_{i}.mp4', 'tmp.mp4')
    for i in range(0, 15):
        exp3(f'resources/009/raw_720p/video009_720p_{i}.mp4', f'resources/009/exp3/video009_720p_{i}_exp3.mp4')
    pass
