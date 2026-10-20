"""
Smoke test: train the train.sh model for 1 epoch on a small QM9 subset, then run evaluate.py end to end.
Both scripts run as real subprocesses, on the CPU, in a temporary folder. Takes about a minute.

Run from the repository root:  python -m unittest tests.test_smoke
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data', 'qm9_relgcn_kekulized_ggnp.npz')


def run(script, *script_args):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable, os.path.join(REPO, script)] + list(script_args), cwd=REPO,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    if result.returncode != 0:
        raise AssertionError('{} failed:\n{}\n{}'.format(script, result.stdout[-3000:], result.stderr[-3000:]))
    return result.stdout


@unittest.skipUnless(os.path.exists(DATA), 'QM9 .npz not found')
class SmokeTest(unittest.TestCase):

    def test_train_one_epoch_then_evaluate(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            # Model settings as train.sh, on 500 training molecules
            run('train_mle.py', '--device', 'cpu', '--subset', '500', '--epochs', '1', '--batch_size', '64',
                '--num_node_masks', '9', '--num_channel_masks', '9',
                '--num_node_coupling', '36', '--num_channel_coupling', '27', '--apply_batch_norm',
                '--num_gen', '5', '--max_resample', '20',
                '--model_save_dir', tmp, '--gen_path', os.path.join(tmp, 'mols'),
                '--img_dir', os.path.join(tmp, 'img'))

            checkpoint_path = os.path.join(tmp, 'epoch1-mle.ckpt')
            checkpoint = torch.load(checkpoint_path)
            self.assertEqual(sorted(checkpoint), ['epoch', 'hyperparams', 'model', 'optimizer', 'train_args'])
            self.assertEqual(checkpoint['epoch'], 1)

            with open(os.path.join(tmp, 'losses.csv')) as f:
                lines = f.read().splitlines()
            self.assertEqual(lines[0], 'epoch,train_loss,valid_nll')
            self.assertEqual(len(lines), 2)
            epoch, train_loss, valid_nll = lines[1].split(',')
            self.assertEqual(int(epoch), 1)
            self.assertTrue(math.isfinite(float(train_loss)) and math.isfinite(float(valid_nll)))

            n = 200
            run('evaluate.py', '--checkpoint', checkpoint_path, '--n', str(n), '--batch_size', '100',
                '--device', 'cpu')
            name = os.path.join(tmp, 'eval_epoch1-mle_n{}_t0.8_s0'.format(n))
            with open(name + '.json') as f:
                results = json.load(f)
            self.assertEqual(sum(results['counts'].values()), n)
            self.assertEqual(results['valid']['count'],
                             results['counts']['valid'] + results['counts']['valid_disconnected'])
            self.assertEqual(results['valid_connected']['count'], results['counts']['valid'])
            self.assertEqual(results['training_split']['molecules'], 500)
            for key in ['checkpoint', 'n', 'temp', 'seed', 'git_commit']:
                self.assertIn(key, results)
            with open(name + '_smiles.txt') as f:
                self.assertEqual(len(f.read().splitlines()), n + 1)  # header + one line per sample


if __name__ == '__main__':
    unittest.main()
