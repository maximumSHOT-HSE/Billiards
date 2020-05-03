from tkinter import *
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
from highlight_table import highlight_table_on_frame
from copy import deepcopy


class Application:

    def __init__(self, width, height):
        self.root = Tk()
        self.root.title('Table recognition ds helper')
        self.root.attributes('-fullscreen', True)
        self.root.bind('q', self.close_button_clicked)
        self.root.bind('d', self.next_button_clicked)

        # configure draw area
        self.canvas = Canvas(self.root, bg='white')
        self.canvas.pack(fill=BOTH, expand=YES)
        self.canvas.bind('<B1-Motion>', self.rmouse_clicked)
        self.d = 0
        self.img = None

        self.polygon_vertices = np.array([[width // 3, height // 3], [2 * width // 3, height // 3],
                                          [2 * width // 3, 2 * height // 3], [width // 3, 2 * height // 3]])
        self.draw_polygon()

    def set_window_size(self, width, height):
        self.root.geometry(f'{width}x{height}')

    def find_nearest_vertex_id(self, x, y):
        p = np.array([x, y])
        best_id = 0
        best_dist = np.inf
        for id, q in enumerate(self.polygon_vertices):
            tmp_dist = np.linalg.norm(q - p)
            if tmp_dist < best_dist:
                best_dist, best_id = tmp_dist, id
        return best_id

    def rmouse_clicked(self, event):
        id = self.find_nearest_vertex_id(event.x, event.y)
        self.polygon_vertices[id] = np.array([event.x, event.y])
        self.draw_polygon()

    def draw_polygon(self):
        self.canvas.delete('all')

        np_img = cv2.imread('42.png')
        self.img = ImageTk.PhotoImage(Image.fromarray(np_img[:, :, ::-1]))

        h, w = np_img.shape[: 2]
        self.d = int(0.25 * max(h, w))

        self.canvas.create_image(self.d, self.d, image=self.img, anchor=NW)
        self.set_window_size(2 * self.d + w, 2 * self.d + h)

        self.canvas.create_polygon(self.polygon_vertices.reshape(-1).tolist(), fill='', outline='green', width=2.5)

    def close_button_clicked(self, event=None):
        self.root.quit()

    def get_relative_hull(self):
        hull = []
        i = 0
        for j, q in enumerate(self.polygon_vertices):
            if q[0] < self.polygon_vertices[i][0] or \
                    (q[0] == self.polygon_vertices[i][0] and q[1] < self.polygon_vertices[i][1]):
                i = j
        v = np.array([self.d, self.d])
        n = len(self.polygon_vertices)
        for iter in range(n):
            j = (iter + i) % n
            hull.append(self.polygon_vertices[j] - v)
        return np.array(hull)

    def next_button_clicked(self, event=None):
        frame = cv2.imread('42.png')
        highlight_table_on_frame(frame, self.get_relative_hull())
        print('SAVE and move to the next one')

    def start(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = Application(1080, 720)
    app.start()
