#!/bin/bash

python train.py \
    --data_dir data/inaturalist_12K \
    --epochs 10 \
    --batch_size 64 \
    --lr 0.0001 \
    --activation mish \
    --dropout 0.4 \
    --dense_neurons 512 \
    --optimizer adam \
    --weight_decay 0.001 \
    --filter_sizes 3 3 3 3 3 \
    --num_filters 32 64 128 128 256 \
    --batch_norm \
    --use_scheduler \
    --scheduler_patience 3 \
    --early_stopping_patience 7 \
    --use_wandb \
    --wandb_project sample_img

