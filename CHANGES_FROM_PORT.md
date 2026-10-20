# Changes to the PyTorch GraphNVP port

Starting point: the PyTorch port [hlzhang109/PyTorch-GraphNVP](https://github.com/hlzhang109/PyTorch-GraphNVP).
Reference: the official Chainer code [pfnet-research/graph-nvp](https://github.com/pfnet-research/graph-nvp).
Goal: make the port behave like the official GraphNVP, then evaluate it with well-defined metrics.
Each change is its own git commit (hash in the last column).

## 1. Model and training: changes that affect results

| # | File(s) | Change | Why (port vs official) | Commit |
|---|---|---|---|---|
| 1 | `graph_nvp/mlp.py`, `graph_nvp/rgcn.py`, `train_mle.py` | ReLU between MLP layers; ReLU applied in the relational graph convolution; official QM9 R-GCN shape (embedding 5→8, one relational conv 8→64 + ReLU, gated readout to 128 without bias); QM9 hidden sizes [128, 64] | The port's MLP had no activations, its R-GCN never applied its ReLU, and its R-GCN had a different shape | `126bfbb` |
| 2 | `graph_nvp/coupling.py` | Last linear layer of every coupling network initialized to weight 1e-10, bias 0 | Official `initialW=1e-10`: each coupling starts as (nearly) the identity. The port used random init | `496f36f` |
| 3 | `train_mle.py` | `model.train()` at the start of every epoch | The port called `model.eval()` for generation after each epoch and never switched back, so BatchNorm stopped updating from epoch 2 | `1211e03` |
| 4 | `graph_nvp/nvp_model.py`, `train_mle.py` | Training: uniform noise on A is added inside the model *after* the X-couplings, only in training mode. Generation: A is discretized to one-hot (symmetrize → softmax → `floor(A / max)`) before X is decoded | As official. The port added the A noise *before* the X-couplings (so they saw a noisy A) and decoded X from a soft A | `fb56a02` |
| 5 | `utils/argparser.py`, `train_mle.py` | Adam, lr 1e-3, betas (0.9, 0.999), constant learning rate | As official (Chainer's default Adam). The port used betas (0.5, 0.9) and a per-batch `ReduceLROnPlateau` | `c315d2a` |
| 6 | `train.sh` (+ comments in `utils/argparser.py`, `graph_nvp/nvp_model.py`) | 36 adjacency couplings, 27 node-feature couplings, 9/9 masks | Official `scripts/train_qm9.sh`; the port's script used 18/14. Naming kept from the official code: `num_node_coupling` = A-couplings, `num_channel_coupling` = X-couplings | `d754ff3` |
| 7 | `utils/argparser.py`, `graph_nvp/nvp_model.py`, `train.sh`, `train_mle.py` | On/off flags use `store_true`; the model always uses additive couplings (no affine path) | With `type=bool` any value, even `False`, switched a flag on. The port's `AffineAdjCoupling.forward` returns its input unchanged; the official QM9/ZINC runs use additive couplings | `5e5e7fb` |
| 8 | `train_mle.py`, `utils/data_utils.py` | Chainer data pipeline replaced by a PyTorch `Dataset` + `DataLoader(shuffle=True)`; same `.npz` file and the same train/validation split | The port replayed the same batch order every epoch; the official iterator reshuffles. Also removes the Chainer dependency | `7c38ef3` |
| 9 | `train_mle.py` | Resume runs epochs `resume_epoch + 1 … epochs` | The port's `range(1 + start, epochs + 1 − start)` skipped epochs, or trained none at all when resuming at or after the halfway point | `c915463` |
| 10 | `train.sh`, `train_mle.py`, `utils/argparser.py` | 60 epochs, checkpoint every 10 (new `--save_epochs`, last epoch always saved) | Official `scripts/train_qm9.sh` (`-x 60 --save_epochs 10`); the port's script trained 10 epochs | `b197541` |
| 11 | `train_mle.py` | ZINC hidden sizes [256, 64] | Official `train_model.py`; the port had them reversed, [64, 256]. Only affects ZINC runs | `TBD-1` |

## 2. Training infrastructure: no effect on the model or the optimization

| File(s) | Change | Commit |
|---|---|---|
| `train_mle.py` | Checkpoints hold model weights, Adam state and epoch, and `--resume` restores all three, as the official Chainer snapshot does. Old weights-only checkpoints still load, with a warning | `02473b1` |
| `train_mle.py`, `graph_nvp/nvp_model.py` | Checkpoints also store the model hyperparameters and training arguments; batch norm is taken from the hyperparameters, not the command line | `7354a01` |
| `train_mle.py` | Per epoch, `losses.csv` in the checkpoint folder: mean training loss, and validation NLL (same loss, eval mode, no dequantization noise, no gradients) | `cd9bd01` |
| `train_mle.py`, `utils/argparser.py` | `--subset n` for quick tests (first n training / n//10 validation molecules; default: all) | `ec6d95b` |
| `utils/sascorer.py` | `rdkit.six` (removed from RDKit) replaced by `dict.items()` | `8bc5f6e` |
| `graph_nvp/rgcn.py`, `utils/argparser.py` | Removed the unused `self.weight` parameter of `RelationGraphConvolution` and the unused `--num_gcn_layer` flag. The model's outputs don't depend on them; the QM9 parameter count drops from 6,201,127 to 6,145,831, and for a given seed the random initial weights differ (one tensor fewer is initialized) | `TBD-1` |

## 3. Evaluation (`evaluate.py`, commit `2277aa3`)

The port's own metrics in `train_mle.py` are not used. In those metrics, validity was always reported as 1.0 once 100 valid molecules turned up in at most 200 tries, disconnected molecules counted as valid, 1-atom molecules were silently dropped, and novelty was measured against the whole dataset including validation, with kekulé and aromatic SMILES mixed.

`evaluate.py` samples n molecules (default 10,000) at a fixed temperature (default 0.8) and seed, with z ~ N(0, (temp·σ)²) where σ² = exp(learned ln_var), the same convention as the official code. Each sample is classified exactly once:

- **too small:** 0 or 1 atoms. Counted as **invalid**.
- **invalid:** RDKit cannot sanitize it.
- **valid, disconnected:** valid, but made of several fragments.
- **valid:** valid and one connected molecule.

The metrics are defined as follows:

- **Validity:** valid / all n samples. Reported twice: including disconnected molecules (`valid`) and excluding them (`valid_connected`). The official GraphNVP code uses the connected version.
- **Uniqueness:** unique valid / valid.
- **Novelty:** valid molecules not in the **training split** (validation excluded) / valid. Also reported over unique molecules (`novelty_unique`), the way the official `generate.py` computes it.
- **Canonicalization:** generated and training molecules are both rebuilt from their tensors with the same `construct_mol`, and both are canonicalized with the same RDKit round trip (`MolFromSmiles(MolToSmiles(mol))`, then non-isomeric canonical SMILES). Check: real training molecules evaluated as if generated are 100% found in the training set, apart from 1-atom molecules, which count as invalid by definition.
- **Output:** a JSON file with all counts and rates, the checkpoint path, n, temperature, seed and git commit, plus a text file listing every sample's SMILES and status.
- **Cache:** the canonical training SMILES are cached in `data/cache/`, keyed on the `.npz` content, the training indices and the atom list (commit `TBD-2`). Cached and recomputed results are identical.
- **Temperature sweep:** `sweep_temperature.py` runs `evaluate.py` on one checkpoint at temperatures 0.6, 0.7, 0.8, 0.85, 0.9 and 1.0 and writes one combined CSV (commit `TBD-3`). All temperatures use the same seed, so they scale the same latent samples.

## 4. Checked and found identical (not changed)

- **Loss.** Both repos use `(nll_x + nll_adj) / 2`, each part being the Gaussian NLL of z minus the log-determinant, divided by the number of dimensions. They agree exactly on random inputs.
  - Both repos subtract the dimension count from the log-determinant, which adds a constant **+1** to the loss. This doesn't affect training, but the loss values are not standard nats per dimension.
  - With additive couplings the log-determinant is 0 in both repos.
- **Sampling temperature convention** (see section 3).
- **Dequantization of X:** uniform noise U(0, 0.9) in both repos. The official code adds it inside the model and the port adds it in the training loop, which gives the same result.

## 5. Remaining differences and caveats

- **Learned prior variance:** `ln_var` is initialized to 1e-5 in the port and to 0 in the official code. The difference is negligible.
- **Left as they were** (no effect on QM9 runs):
  - The `-lr_decay_*` flags are unused.
  - The affine coupling classes are still in `coupling.py` but are no longer used.
- **ZINC can't be trained yet.** `nvp_model.py` asserts `num_relations + 1 == num_features`, which holds for QM9 only by coincidence (4 + 1 = 5); for ZINC (10 atom features) it fails, so the ZINC model can't be built. The official code has no such check.
- **Resumed runs are not bit-identical to uninterrupted runs.** The random-number state (shuffle order, noise) is not stored in checkpoints.
- **Chainer is still needed to create the dataset.** `data/download_data.py` needs chainer-chemistry; the existing `.npz` files were made with it.
- **The tensors don't store formal charges.** Charged molecules can't be rebuilt from their tensors: 134 of the 120,803 QM9 training molecules. They therefore never match a generated molecule.
- **Finding about the official evaluation.** The official `generate.py` builds the training SMILES from *unsanitized* molecules (kekulé form) but the generated SMILES from *sanitized* ones (aromatic form). On 2,000 real QM9 training molecules, 331 (16.6%, every one with an aromatic ring) would be counted as novel. The official novelty numbers are therefore likely overestimated. `evaluate.py` avoids this.

## 6. Tests (`python -m unittest discover tests`)

- **`tests/test_invertibility.py`** (`7ec7eaf`) uses the official QM9 model in eval mode, with random last-layer weights so the test isn't trivial. On a real QM9 batch it checks that:
  - the X-couplings invert, to below 1e-4;
  - the A-couplings invert on dequantized A, to below 1e-4;
  - `reverse(forward(A, X))` recovers the one-hot A exactly and X to below 1e-4.
- **`tests/test_smoke.py`** (`8d0949b`) trains the `train.sh` model for 1 epoch on 500 molecules, checks the checkpoint and `losses.csv`, then runs `evaluate.py` and checks its outputs.
