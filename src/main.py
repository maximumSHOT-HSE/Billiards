import argparse
import sys

from src.image_operations import read_image, calc_circles

if __name__ == '__main__':
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser(
        description='Red an black circles counter'
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to the image in .png format'
    )

    args = parser.parse_args()
    file_path = args.file
    image = read_image(file_path)
    c_red, c_black = calc_circles(image)
    print(f'{c_red} {c_black}')
