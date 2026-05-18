import unittest

import torch

from rfo_demo.models import FNO1d, FNO2d, FNO3d


class ModelShapeTest(unittest.TestCase):
    def test_fno1d_shape(self):
        model = FNO1d(4, 8)
        x = torch.randn(2, 16)
        t = torch.randn(2, 16)
        y = model(x, t)
        self.assertEqual(tuple(y.shape), (2, 16))

    def test_fno2d_shape(self):
        model = FNO2d(4, 4, 8)
        x = torch.randn(2, 8, 8)
        t = torch.randn(2, 8, 8)
        y = model(x, t)
        self.assertEqual(tuple(y.shape), (2, 8, 8))

    def test_fno3d_shape(self):
        model = FNO3d(4, 4, 3, 8)
        x = torch.randn(2, 8, 8, 4, 1)
        t = torch.randn(2, 8, 8, 4, 1)
        y = model(x, t)
        self.assertEqual(tuple(y.shape), (2, 8, 8, 4, 1))


if __name__ == "__main__":
    unittest.main()
