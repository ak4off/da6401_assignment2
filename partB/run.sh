#!/bin/bash

python3 the_trainer.py \
    --data_dir data/inaturalist_12K \
    --epochs 3 \
    --batch_size 64 \
    --lr 0.001 \
    --dropout 0.1 \
    --weight_decay 0.0001 \
    --data_aug
    # --use_wandb \
    # --wandb_project cnn_from_scratch_project \
    # --run_name best_cnn_tryout
