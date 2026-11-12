"""
Dequantization placement and gradient flow, on a real QM9 batch.

Run from the repository root:  python -m unittest tests.test_model_behaviour
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


def qm9_model():
    """The official QM9 model, as train.sh"""
    hp = Hyperparameters(9, 4, 5, num_masks={'node': 9, 'channel': 9}, mask_size={'node': 15, 'channel': -1},
                         num_coupling={'node': 36, 'channel': 27}, batch_norm=True,
                         additive_transformations=True, mlp_channels=[256, 256],
                         gnn_channels={'gcn': [8, 64], 'hidden': [128, 64]})
    return GraphNvpModel(hp)


def randomize_last_layers(model):
    # A fresh model is the identity (near-zero init); random last layers make every coupling act
    for t in model.transforms:
        nn.init.normal_(t.lin2.weight if hasattr(t, 'lin2') else t.lin.weight, std=0.05)


@unittest.skipUnless(os.path.exists(DATA), 'QM9 .npz not found')
class DequantizationTest(unittest.TestCase):
    """A is dequantized after the X-couplings (they see the clean one-hot A), as in the official code"""

    @classmethod
    def setUpClass(cls):
        args.device = 'cpu'
        torch.manual_seed(0)
        dataset = NumpyTupleDataset(DATA, transform_qm9.transform_fn)
        X, A, _ = next(iter(DataLoader(dataset, batch_size=64, shuffle=True)))
        cls.X, cls.A = X.float(), A.float()
        cls.X_deq = cls.X + 0.9 * torch.rand(cls.X.shape)
        cls.noise = 0.9 * torch.rand(cls.A.shape)
        cls.model = qm9_model()
        randomize_last_layers(cls.model)

    def forward(self, adj, x, **kwargs):
        with torch.no_grad():
            (z_x, z_adj), _ = self.model(adj, x, **kwargs)
        return z_x, z_adj

    def test_noise_enters_after_the_x_couplings(self):
        self.model.eval()
        zx_clean, _ = self.forward(self.A, self.X_deq)
        zx_noise, za_noise = self.forward(self.A, self.X_deq, adj_noise=self.noise)
        zx_noisy_input, za_noisy_input = self.forward(self.A + self.noise, self.X_deq)
        # The X-couplings do depend on A, so the next check is meaningful ...
        self.assertFalse(torch.equal(zx_noisy_input, zx_clean))
        # ... and with adj_noise they still saw the clean A
        self.assertTrue(torch.equal(zx_noise, zx_clean))
        # The A-couplings saw exactly A + noise
        self.assertTrue(torch.equal(za_noise, za_noisy_input))

    def test_training_mode_adds_fresh_noise_to_a_only(self):
        self.model.train()
        zx1, za1 = self.forward(self.A, self.X_deq)
        zx2, za2 = self.forward(self.A, self.X_deq)
        self.assertTrue(torch.equal(zx1, zx2), 'X part changed: A noise reached the X-couplings')
        self.assertFalse(torch.equal(za1, za2), 'no fresh A noise in training mode')

    def test_eval_mode_adds_no_noise(self):
        self.model.eval()
        zx1, za1 = self.forward(self.A, self.X_deq)
        zx2, za2 = self.forward(self.A, self.X_deq)
        _, za_zero = self.forward(self.A, self.X_deq, adj_noise=torch.zeros_like(self.A))
        self.assertTrue(torch.equal(zx1, zx2) and torch.equal(za1, za2))
        self.assertTrue(torch.equal(za1, za_zero))

    def test_training_noise_is_uniform_0_to_0_9(self):
        # A fresh model is the identity up to ~1e-9, so z_adj - A is the added noise itself
        fresh = qm9_model()
        fresh.train()
        with torch.no_grad():
            (_, z_adj), _ = fresh(self.A, self.X_deq)
        noise = z_adj - self.A.view(len(self.A), -1)
        self.assertGreater(noise.min().item(), -1e-6)
        self.assertLess(noise.max().item(), 0.9 + 1e-6)
        self.assertAlmostEqual(noise.mean().item(), 0.45, delta=0.01)


@unittest.skipUnless(os.path.exists(DATA), 'QM9 .npz not found')
class GradientFlowTest(unittest.TestCase):
    """Every parameter gets a gradient and is updated (no detached or unused parts)"""

    def test_all_parameters_are_trained(self):
        args.device = 'cpu'
        torch.manual_seed(0)
        dataset = NumpyTupleDataset(DATA, transform_qm9.transform_fn)
        X, A, _ = next(iter(DataLoader(dataset, batch_size=64, shuffle=True)))
        X, A = X.float(), A.float()
        model = qm9_model()
        model.train()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            z, sum_log_det_jacs = model(A, X + 0.9 * torch.rand(X.shape))
            loss = model.log_prob(z, sum_log_det_jacs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        no_grad = [name for name, p in model.named_parameters() if p.grad is None]
        not_finite = [name for name, p in model.named_parameters()
                      if p.grad is not None and not torch.isfinite(p.grad).all()]
        unchanged = [name for name, p in model.named_parameters() if torch.equal(p.detach(), before[name])]
        self.assertEqual(no_grad, [], 'parameters without gradient')
        self.assertEqual(not_finite, [], 'parameters with non-finite gradient')
        self.assertEqual(unchanged, [], 'parameters not updated')


if __name__ == '__main__':
    unittest.main()
