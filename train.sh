#!/usr/bin/env bash
# Layer counts and masks as in the official scripts/train_qm9.sh: 36 adjacency (A) couplings,
# 27 node-feature (X) couplings, 9/9 masks. Naming quirk kept from the official code:
# --num_node_coupling counts the A-couplings, --num_channel_coupling counts the X-couplings.
CUDA_VISIBLE_DEVICES=1 python train_mle.py --batch_size 256 \
    --data_file qm9_relgcn_kekulized_ggnp.npz \
    --epochs 10 \
    --device cuda \
    --num_node_masks 9 \
    --num_channel_masks 9 \
    --num_node_coupling 36 \
    --num_channel_coupling 27 \
    --apply_batch_norm \
    --node_mask_size 15 \
    --mode train \
    --num_gen 100 \
    --show_loss_step 5 \
    --gen_path ./results/qm9_mols \
    --model_save_dir ./saved_qm9_models \
    --img_dir=./results/qm9_img
