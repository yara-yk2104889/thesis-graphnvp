"""
Evaluate a trained GraphNVP checkpoint: sample molecules and report validity, uniqueness and novelty.

    python evaluate.py --checkpoint saved_qm9_models/epoch60-mle.ckpt --n 10000 --temp 0.8 --seed 0

Definitions (every sample is classified once):
- too_small:          0 or 1 atoms                                   -> invalid
- invalid:            RDKit cannot parse/sanitize the molecule       -> invalid
- valid_disconnected: valid, but several fragments ('.' in SMILES)
- valid:              valid and one connected molecule
Validity is reported over ALL n samples, both including disconnected molecules ("valid")
and excluding them ("valid_connected", the official GraphNVP definition).
Uniqueness = unique / valid. Novelty = valid molecules whose SMILES is not in the TRAINING
split (validation molecules excluded); novelty_unique = the same over unique molecules (as
the official generate.py reports it). Generated and training molecules are built from their
tensors with the same construct_mol and canonicalized with the same function.

Writes <out_dir>/eval_<checkpoint>_n<n>_t<temp>_s<seed>.json and ..._smiles.txt.
The canonical training SMILES are cached in data/cache/ (keyed on the .npz content and the split),
so only the first evaluation for a dataset spends minutes on them.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from os.path import join as joinpath

parser = argparse.ArgumentParser(description='Evaluate a GraphNVP checkpoint')
parser.add_argument('--checkpoint', type=str, required=True, help='.ckpt file saved by train_mle.py')
parser.add_argument('--n', type=int, default=10000, help='number of molecules to sample')
parser.add_argument('--temp', type=float, default=0.8, help='temperature: z ~ N(0, (temp * sigma)^2)')
parser.add_argument('--seed', type=int, default=0, help='random seed for sampling')
parser.add_argument('--batch_size', type=int, default=1000, help='molecules decoded per batch')
parser.add_argument('--out_dir', type=str, default=None, help='output folder (default: the checkpoint folder)')
parser.add_argument('--device', type=str, default=None, help='cpu or cuda (default: cuda if available)')
parser.add_argument('--cache_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  'data', 'cache'),
                    help='where the canonical training SMILES are cached')
parser.add_argument('--no_cache', action='store_true', help='recompute the training SMILES, do not read or write the cache')
eval_args = parser.parse_args()

# The model code reads utils.argparser.args, which parses sys.argv on import: give it the defaults
sys.argv = sys.argv[:1]

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import Subset
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.nvp_model import GraphNvpModel
from utils.argparser import args
from utils.data_utils import NumpyTupleDataset, construct_mol

RDLogger.DisableLog('rdApp.*')  # RDKit errors for invalid samples are expected; they are counted instead
REPO = os.path.dirname(os.path.abspath(__file__))
CACHE_VERSION = 1  # bump when construct_mol or canonical_smiles change, so old caches are not reused


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu')
    if 'hyperparams' not in checkpoint:
        sys.exit('{} has no model settings (saved before checkpoints stored them); '
                 'retrain or re-save it with the current train_mle.py'.format(path))
    hyperparams = Hyperparameters()
    for key, value in checkpoint['hyperparams'].items():
        setattr(hyperparams, key, value)
    model = GraphNvpModel(hyperparams).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def canonical_smiles(mol):
    """
    Canonical SMILES of a molecule built by construct_mol, or None if RDKit cannot sanitize it.
    Same round trip as the official valid_mol; used for generated AND training molecules.
    """
    try:
        sanitized = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    except Exception:
        return None
    if sanitized is None:
        return None
    return Chem.MolToSmiles(sanitized, isomericSmiles=False)


def classify(mol):
    """Returns (status, canonical SMILES or None)"""
    if mol.GetNumAtoms() < 2:
        return 'too_small', None
    smiles = canonical_smiles(mol)
    if smiles is None:
        return 'invalid', None
    return ('valid_disconnected' if '.' in smiles else 'valid'), smiles


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def training_split_smiles(train_args, atomic_num_list, cache_dir=None):
    """
    Canonical SMILES of the training split used by train_mle.py (same split and --subset).
    Cached in cache_dir, keyed on the .npz content, the training indices and the atom list.
    Returns (smiles set, number of training molecules, number not rebuildable, cache file or None, cache hit)
    """
    data_name = train_args['data_name']
    if data_name == 'qm9':
        from data import transform_qm9
        transform_fn, valid_idx = transform_qm9.transform_fn, transform_qm9.get_val_ids()
    else:
        from data import transform_zinc250k
        transform_fn, valid_idx = transform_zinc250k.transform_fn_zinc250k, transform_zinc250k.get_val_ids()
    npz_path = joinpath(train_args['data_dir'], train_args['data_file'])
    with np.load(npz_path) as f:
        n_molecules = len(f['arr_0'])
    if len(valid_idx) > 0:
        valid_set = set(valid_idx)
        train_idx = [t for t in range(n_molecules) if t not in valid_set]
    else:
        order = np.random.RandomState(train_args['seed']).permutation(n_molecules)
        train_idx = [int(t) for t in order[:int(n_molecules * 0.8)]]
    if train_args.get('subset', 0) > 0:
        train_idx = train_idx[:train_args['subset']]

    cache_path = None
    if cache_dir is not None:
        key = hashlib.sha256(json.dumps({'version': CACHE_VERSION, 'npz_sha256': file_sha256(npz_path),
                                         'train_idx': train_idx, 'atomic_num_list': atomic_num_list}
                                        ).encode()).hexdigest()
        cache_path = joinpath(cache_dir, 'train_smiles_{}_{}.json'.format(data_name, key[:16]))
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if cached['key'] == key:
                return set(cached['smiles']), cached['n_train'], cached['not_rebuilt'], cache_path, True

    print('canonicalizing the training split...')
    train = Subset(NumpyTupleDataset(npz_path, transform_fn), train_idx)
    smiles, not_rebuilt = set(), 0
    for X, A, _ in train:
        s = canonical_smiles(construct_mol(torch.as_tensor(X), torch.as_tensor(A), atomic_num_list))
        if s is None:
            not_rebuilt += 1
        else:
            smiles.add(s)

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'key': key, 'npz': os.path.abspath(npz_path), 'n_train': len(train_idx),
                       'not_rebuilt': not_rebuilt, 'smiles': sorted(smiles)}, f)
        os.replace(tmp, cache_path)  # atomic: a half-written cache is never read
    return smiles, len(train_idx), not_rebuilt, cache_path, False


def rates(smiles_list, train_smiles, n_samples):
    count = len(smiles_list)
    unique = set(smiles_list)
    novel = [s for s in smiles_list if s not in train_smiles]
    novel_unique = unique - train_smiles

    def ratio(a, b):
        return a / b if b > 0 else None
    return {'count': count, 'validity': ratio(count, n_samples),
            'unique': len(unique), 'uniqueness': ratio(len(unique), count),
            'novel': len(novel), 'novelty': ratio(len(novel), count),
            'novel_unique': len(novel_unique), 'novelty_unique': ratio(len(novel_unique), len(unique))}


def git_state():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip() != ''
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main():
    device = eval_args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device  # coupling layers read the device from utils.argparser.args
    model, checkpoint = load_model(eval_args.checkpoint, device)
    train_args = checkpoint['train_args']
    atomic_num_list = train_args['atomic_num_list']

    # Sample on the CPU generator so the same seed gives the same z on CPU and GPU
    generator = torch.Generator().manual_seed(eval_args.seed)
    sigma = eval_args.temp * math.sqrt(math.exp(model.prior_ln_var.item()))
    statuses, smiles = [], []
    with torch.no_grad():
        for start in range(0, eval_args.n, eval_args.batch_size):
            batch = min(eval_args.batch_size, eval_args.n - start)
            z = torch.randn(batch, model.x_size + model.adj_size, generator=generator) * sigma
            adj, x = model.reverse(z.to(device), model.x_size)
            for x_elem, adj_elem in zip(x.cpu(), adj.cpu()):
                mol = construct_mol(x_elem, adj_elem, atomic_num_list)
                status, s = classify(mol)
                statuses.append(status)
                # For invalid samples keep RDKit's raw SMILES of the unsanitized graph, for inspection
                smiles.append(s if s is not None else Chem.MolToSmiles(mol))
            print('sampled {}/{}'.format(start + batch, eval_args.n))

    cache_dir = None if eval_args.no_cache else eval_args.cache_dir
    train_smiles, train_size, not_rebuilt, cache_path, cache_hit = training_split_smiles(
        train_args, atomic_num_list, cache_dir)
    if cache_hit:
        print('training SMILES loaded from cache', cache_path)

    valid_smiles = [s for s, st in zip(smiles, statuses) if st in ('valid', 'valid_disconnected')]
    connected_smiles = [s for s, st in zip(smiles, statuses) if st == 'valid']
    commit, dirty = git_state()
    results = {
        'checkpoint': os.path.abspath(eval_args.checkpoint),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'n': eval_args.n, 'temp': eval_args.temp, 'seed': eval_args.seed,
        'git_commit': commit, 'git_uncommitted_changes': dirty,
        'counts': {status: statuses.count(status)
                   for status in ['valid', 'valid_disconnected', 'invalid', 'too_small']},
        'valid': rates(valid_smiles, train_smiles, eval_args.n),
        'valid_connected': rates(connected_smiles, train_smiles, eval_args.n),
        'training_split': {'molecules': train_size, 'unique_canonical_smiles': len(train_smiles),
                           'could_not_be_rebuilt': not_rebuilt, 'cache_file': cache_path,
                           'from_cache': cache_hit},
    }

    out_dir = eval_args.out_dir or os.path.dirname(os.path.abspath(eval_args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    name = 'eval_{}_n{}_t{}_s{}'.format(os.path.splitext(os.path.basename(eval_args.checkpoint))[0],
                                        eval_args.n, eval_args.temp, eval_args.seed)
    with open(joinpath(out_dir, name + '.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(joinpath(out_dir, name + '_smiles.txt'), 'w') as f:
        f.write('smiles\tstatus\n')
        for s, status in zip(smiles, statuses):
            f.write('{}\t{}\n'.format(s, status))

    for key in ['valid', 'valid_connected']:
        r = results[key]
        fmt = lambda v: 'n/a' if v is None else '%.4f' % v
        print('%-16s validity %s | uniqueness %s | novelty %s | novelty (unique) %s' % (
            key, fmt(r['validity']), fmt(r['uniqueness']), fmt(r['novelty']), fmt(r['novelty_unique'])))
    print('counts:', results['counts'])
    print('saved', joinpath(out_dir, name + '.json'))


if __name__ == '__main__':
    main()
