import argparse
import math
from collections import deque
from copy import deepcopy

import cv2
import numpy as np
from scipy.spatial import ConvexHull


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finds table polygon for each frame'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to the video to be processed'
    )
    parser.add_argument(
        '--layout',
        type=str,
        required=True,
        help='Path to the output fill for storing layout'
    )
    return parser.parse_args()


# finds mask of pixels which are close enough to the mean color of the table
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


# removes vertex which produces the largest angle until it greater than threshold
def remove_big_angles_from_hull(hull, angle_threshold=np.pi * 160 / 180):
    n = len(hull)

    def get_angle(i):
        prv = (i - 1 + n) % n
        nxt = (i + 1) % n

        v1 = hull[nxt] - hull[i]
        v2 = hull[prv] - hull[i]

        return math.acos(np.sum(v1 * v2) / np.linalg.norm(v1) / np.linalg.norm(v2))

    while True:
        mx_angle_id = 0
        mx_angle = 0
        for i in range(n):
            ith_angle = get_angle(i)
            if ith_angle > mx_angle:
                mx_angle, mx_angle_id = ith_angle, i
        if mx_angle < angle_threshold:
            break
        hull = np.concatenate((hull[: mx_angle_id], hull[mx_angle_id + 1: ]))
        n -= 1

    return hull


def take_longest_sides_from_hull(hull, k):
    n = len(hull)
    assert n >= k
    ids = list(range(n))
    ids.sort(key=lambda i: np.linalg.norm(hull[i] - hull[(i + 1) % n]), reverse=True)
    ids = list(sorted(ids[: k]))

    khull = []
    for it in range(k):
        i1 = ids[it]
        i2 = (i1 + 1) % n
        j1 = ids[(it + 1) % k]
        j2 = (j1 + 1) % n

        v1 = hull[i1] - hull[i2]
        v2 = hull[j1] - hull[j2]

        assert np.linalg.norm(v1) > 0
        assert np.linalg.norm(v2) > 0

        t = (
            (hull[i1][1] - hull[j1][1]) * v2[0] -
            (hull[i1][0] - hull[j1][0]) * v2[1]
        ) / (v1[0] * v2[1] - v1[1] * v2[0])

        khull.append(hull[i1] + v1 * t)

    return np.array(khull, dtype=int)


if __name__ == '__main__':
    np.random.seed(42)

    args = parse_args()

    capture = cv2.VideoCapture(args.video)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    with open(args.layout, 'w') as layout_file:
        while capture.isOpened():
            response, frame = capture.read()

            if not response:
                break

            hull = find_table_polygon(deepcopy(frame))
            hull = remove_big_angles_from_hull(hull)
            hull = take_longest_sides_from_hull(hull, 4)
            assert len(hull) == 4
            hull = hull[:, ::-1]

            for x, y in hull:
                layout_file.write(f'{x} {y} ')
            layout_file.write('\n')
