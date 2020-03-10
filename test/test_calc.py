import os
import sys
import unittest

from src.image_operations import read_image, calc_circles


class TestCalcCircles(unittest.TestCase):
    def setUp(self):
        self.root = 'resources'
        sys.setrecursionlimit(10000)

    def test_rotate(self):
        test_images = set(
            file[:-4]
            for file in os.listdir(self.root)
            if file.endswith('.png')
        )
        for test_image in test_images:
            image_path = os.path.join(self.root, f'{test_image}.png')
            ans_file = os.path.join(self.root, f'{test_image}.ans')

            image = read_image(image_path)

            calculated_c_red, calculated_c_black = calc_circles(image)
            with open(ans_file) as ans_fin:
                expected_c_red, expected_c_black = list(map(int, ans_fin.readline().split(' ')))

            self.assertEqual(expected_c_red, calculated_c_red)
            self.assertEqual(expected_c_black, calculated_c_black)
