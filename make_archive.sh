#!/usr/bin/env bash
# Fallback for a machine that cannot reach GitHub: zip the current commit with git archive.
#
#   bash make_archive.sh        (on Windows: in Git Bash)
#
# Writes thesis-graphnvp-<commit>.zip in the repository folder, containing the folder
# thesis-graphnvp-<commit>/. Only committed files are included: no uncommitted changes, no .npz
# data, no checkpoints. The zip also contains a file COMMIT with the full commit ID, which
# evaluate.py records when there is no .git folder. Copy the data separately, e.g.
#
#   scp thesis-graphnvp-<commit>.zip data/qm9_relgcn_kekulized_ggnp.npz panther:
#   ssh panther 'unzip thesis-graphnvp-<commit>.zip && mv qm9_relgcn_kekulized_ggnp.npz thesis-graphnvp-<commit>/data/'
set -euo pipefail
cd "$(dirname "$0")"

full=$(git rev-parse HEAD)
name="thesis-graphnvp-$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Warning: there are uncommitted changes; they are NOT in the zip" >&2
fi
# --prefix does not apply to --add-virtual-file, so COMMIT gets the folder name explicitly
git archive --format=zip --prefix="$name/" --add-virtual-file="$name/COMMIT:$full" -o "$name.zip" HEAD
echo "wrote $name.zip (commit $full)"
