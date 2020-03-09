import numpy as np
from typing import Union, List
import os


# read matrix from the input file
def read_matrix(file_path: str) -> np.ndarray:
    with open(file_path) as fin:
        n, m = list(map(int, fin.readline().split(' ')))
        assert n > 0 and m > 0
        rows = [
            list(map(int, line.split(' ')))
            for line in fin.readlines()
        ]
        assert len(rows) == n
        for row in rows:
            assert len(row) == m
        matrix = np.array(rows)
        return matrix


# writes matrix into given output descriptor
def write_matrix(fout, matrix: np.ndarray):
    n, m = matrix.shape
    fout.write(f'{n} {m}{os.linesep}')
    for row in matrix:
        fout.write(f"{' '.join(map(str, row))}{os.linesep}")


# rotates matrix clockwise by 90 degree
def rotate_matrix(matrix: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    matrix = np.array(matrix)
    if len(matrix.shape) != 2:
        raise Exception(f'Matrix shape should have exactly two dimensions, but found {len(matrix.shape)} dimensions')
    transposed_matrix = matrix.T
    rotated_matrix = transposed_matrix[:, ::-1]
    return rotated_matrix
