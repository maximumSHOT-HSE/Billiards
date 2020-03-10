import matplotlib.pyplot as plt
import numpy as np


# checks whether pixel is red or not
def is_red(pixel: np.ndarray) -> bool:
    return (pixel == np.array([255, 0, 0])).all()


# checks whether pixel is black or not
def is_black(pixel: np.ndarray) -> bool:
    return (pixel == np.array([0, 0, 0])).all()


# checks whether two given pixels are equal or not
def are_equal(pixel_a: np.ndarray, pixel_b: np.ndarray) -> bool:
    return (pixel_a == pixel_b).all()


def dfs(used, image, n, m, i, j, component):
    if used[i, j]:
        return
    component.append((i, j))
    used[i, j] = 1
    for di, dj in [(-1, 0), (+1, 0), (0, -1), (0, +1)]:
        ti, tj = i + di, j + dj
        if ti < 0 or ti >= n or tj < 0 or tj >= m:
            continue
        if are_equal(image[i, j], image[ti, tj]):
            dfs(used, image, n, m, ti, tj, component)


# splits image into components ignoring green pixels
def get_components(image: np.ndarray):
    n, m, _ = image.shape
    used = np.zeros((n, m))
    components = []
    for i in range(n):
        for j in range(m):
            if used[i, j]:
                continue
            component = []
            dfs(used, image, n, m, i, j, component)
            components.append(component)
    return components


def is_rectangle(component):
    _is = np.array([i for i, _ in component])
    min_i = _is.min()
    max_i = _is.max()

    _js = np.array([j for _, j in component])
    min_j = _js.min()
    max_j = _js.max()

    return len(component) == (max_i - min_i + 1) * (max_j - min_j + 1)


# calculates the number of red and black circles on image and returns appropriate counts
def calc_circles(image):
    components = get_components(image)
    c_red = 0
    c_black = 0
    for i, component in enumerate(components):
        color = image[component[0][0], component[0][1]]
        if is_rectangle(component):
            continue
        if is_red(color):
            c_red += 1
        if is_black(color):
            c_black += 1
    return c_red, c_black


# reads the image and returns corresponding two dimensional array of triples in format RGB
def read_image(image_path: str):
    return (plt.imread(image_path, 'png') * 255).astype(dtype='uint8')[..., :3]


# two dimensional array of triples in format RGB to the given path
def save_image(pixels, path):
    plt.imsave(path, np.array(pixels, dtype='uint8'), format='png')
