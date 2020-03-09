import argparse
import sys

from src.maxtrix_operations import rotate_matrix, read_matrix, write_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Two dimensional matrix rotator'
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to the input file'
    )
    args = parser.parse_args()
    file_path = args.file

    matrix = read_matrix(file_path)
    rotated_matrix = rotate_matrix(matrix)
    write_matrix(sys.stdout, rotated_matrix)
