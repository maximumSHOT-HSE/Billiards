import os
import unittest

import numpy as np

from src.maxtrix_operations import read_matrix, rotate_matrix


class TestRotate(unittest.TestCase):
    def setUp(self):
        self.root = 'resources'

    def test_rotate(self):
        test_matrixes = set(
            file[:-3]
            for file in os.listdir(self.root)
            if file.endswith('.in')
        )
        for matrix_path in test_matrixes:
            matrix_in_file = os.path.join(self.root, f'{matrix_path}.in')
            matrix_out_file = os.path.join(self.root, f'{matrix_path}.out')

            matrix = read_matrix(matrix_in_file)
            rotated_matrix = rotate_matrix(matrix)
            expected_matrix = read_matrix(matrix_out_file)

            self.assertEqual(True, np.alltrue(expected_matrix == rotated_matrix))
