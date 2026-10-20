"""
Invertibility of the flow on a real QM9 batch: forward then reverse must give back the input.

Run from the repository root:  python -m unittest tests.test_invertibility
"""
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# utils.argparser parses sys.argv on import; give it only the defaults (not unittest's arguments)
_argv, sys.argv = sys.argv, sys.argv[:1]
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import transform_qm9
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.nvp_model import GraphNvpModel
from utils.argparser import args
from utils.data_utils import NumpyTupleDataset
sys.argv = _argv

DATA = os.path.join(REPO, 'data', 'qm9_relgcn_kekulized_ggnp.npz')


@unittest.skipUnless(os.path.exists(DATA), 'QM9 .npz not found')
class InvertibilityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        args.device = 'cpu'
        torch.manual_seed(0)
        # Official QM9 model (as train.sh)
        hp = Hyperparameters(9, 4, 5, num_masks={'node': 9, 'channel': 9}, mask_size={'node': 15, 'channel': -1},
                             num_coupling={'node': 36, 'channel': 27}, batch_norm=True,
                             additive_transformations=True, mlp_channels=[256, 256],
                             gnn_channels={'gcn': [8, 64], 'hidden': [128, 64]})
        cls.model = GraphNvpModel(hp)
        # A fresh model is the identity (near-zero init), which would make the test trivial:
        # give every coupling's last layer random weights so each layer really changes its input
        for t in cls.model.transforms:
            nn.init.normal_(t.lin2.weight if hasattr(t, 'lin2') else t.lin.weight, std=0.05)
        # Eval mode, as at generation: no A noise, BatchNorm uses fixed (running) statistics
        cls.model.eval()

        dataset = NumpyTupleDataset(DATA, transform_qm9.transform_fn)
        X, A, _ = next(iter(DataLoader(dataset, batch_size=64, shuffle=True)))
        cls.X, cls.A = X.float(), A.float()
        cls.X_deq = cls.X + 0.9 * torch.rand(cls.X.shape)
        cls.A_deq = cls.A + 0.9 * torch.rand(cls.A.shape)

    def test_x_couplings_invert(self):
        model, n_x = self.model, self.model.num_coupling['channel']
        with torch.no_grad():
            h = self.X_deq
            for t in model.transforms[:n_x]:
                h, _ = t(h, self.A)
            self.assertGreater((h - self.X_deq).abs().max().item(), 1e-2, 'couplings did not change X')
            for t in reversed(model.transforms[:n_x]):
                h, _ = t.reverse(h, self.A)
        self.assertLess((h - self.X_deq).abs().max().item(), 1e-4)

    def test_a_couplings_invert(self):
        # On a dequantized (continuous) A, before generation's discretization step
        model, n_x = self.model, self.model.num_coupling['channel']
        with torch.no_grad():
            h = self.A_deq
            for t in model.transforms[n_x:]:
                h, _ = t(h)
            self.assertGreater((h - self.A_deq).abs().max().item(), 1e-2, 'couplings did not change A')
            for t in reversed(model.transforms[n_x:]):
                h, _ = t.reverse(h)
        self.assertLess((h - self.A_deq).abs().max().item(), 1e-4)

    def test_model_round_trip(self):
        # Full model: forward(A, X), then reverse() including the one-hot discretization of A
        model = self.model
        with torch.no_grad():
            (z_x, z_adj), _ = model(self.A, self.X_deq)
            adj, x = model.reverse(torch.cat([z_x, z_adj], dim=1), model.x_size)
        self.assertTrue(torch.equal(adj, self.A), 'one-hot A not recovered exactly')
        self.assertLess((x - self.X_deq).abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
