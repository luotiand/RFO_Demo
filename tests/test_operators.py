import unittest

import torch

from rfo_demo.operators import RectFlow


class RectFlowTest(unittest.TestCase):
    def test_straight_forward_reverse(self):
        rf = RectFlow()
        x0 = torch.tensor([[0.0, 1.0]])
        x1 = torch.tensor([[2.0, 3.0]])
        t = torch.tensor([[0.5, 0.5]])
        xt = rf.straight_process(x0, x1, t)
        self.assertTrue(torch.allclose(xt, torch.tensor([[1.0, 2.0]])))
        score = torch.ones_like(xt)
        self.assertTrue(torch.allclose(rf.forward_process(xt, score, 0.1), torch.tensor([[1.1, 2.1]])))
        self.assertTrue(torch.allclose(rf.reverse_process(xt, score, 0.1), torch.tensor([[0.9, 1.9]])))


if __name__ == "__main__":
    unittest.main()
