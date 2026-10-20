"""
Evaluate one checkpoint at several sampling temperatures and write one combined CSV.

    python sweep_temperature.py --checkpoint saved_qm9_models/epoch60-mle.ckpt

Runs evaluate.py once per temperature with the same --n and --seed. evaluate.py draws
z = N(0, I) * temp * sigma from that seed, so all temperatures scale the same latent samples.
Any other flags (e.g. --device, --batch_size, --cache_dir) are passed on to evaluate.py.
Writes <out_dir>/temp_sweep_<checkpoint>_n<n>_s<seed>.csv, one row per temperature, next to
the per-temperature JSON and SMILES files written by evaluate.py.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
METRICS = ['validity', 'uniqueness', 'novelty', 'novelty_unique', 'count', 'unique', 'novel', 'novel_unique']
STATUSES = ['valid', 'valid_disconnected', 'invalid', 'too_small']

parser = argparse.ArgumentParser(description='Run evaluate.py at several temperatures')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--temps', type=float, nargs='+', default=[0.6, 0.7, 0.8, 0.85, 0.9, 1.0])
parser.add_argument('--n', type=int, default=10000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out_dir', type=str, default=None, help='output folder (default: the checkpoint folder)')
sweep_args, evaluate_args = parser.parse_known_args()

out_dir = sweep_args.out_dir or os.path.dirname(os.path.abspath(sweep_args.checkpoint))
stem = os.path.splitext(os.path.basename(sweep_args.checkpoint))[0]
rows = []
for temp in sweep_args.temps:
    print('=== temperature {}'.format(temp), flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, 'evaluate.py'), '--checkpoint', sweep_args.checkpoint,
                    '--n', str(sweep_args.n), '--temp', str(temp), '--seed', str(sweep_args.seed),
                    '--out_dir', out_dir] + evaluate_args, check=True)
    # Same file name as evaluate.py writes
    json_path = os.path.join(out_dir, 'eval_{}_n{}_t{}_s{}.json'.format(stem, sweep_args.n, temp, sweep_args.seed))
    with open(json_path) as f:
        result = json.load(f)
    row = {'temp': temp, 'n': result['n'], 'seed': result['seed'], 'checkpoint_epoch': result['checkpoint_epoch']}
    for key in ['valid', 'valid_connected']:
        for metric in METRICS:
            row['{}_{}'.format(key, metric)] = result[key][metric]
    for status in STATUSES:
        row['n_' + status] = result['counts'][status]
    row['git_commit'] = result['git_commit']
    row['json'] = os.path.basename(json_path)
    rows.append(row)

csv_path = os.path.join(out_dir, 'temp_sweep_{}_n{}_s{}.csv'.format(stem, sweep_args.n, sweep_args.seed))
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('wrote', csv_path)
