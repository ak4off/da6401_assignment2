### ✅ `wandb_sweep.py`


import wandb
import os
import subprocess

sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [1e-3, 1e-4, 5e-4]},
        'batch_size': {'values': [32, 64]},
        'dropout': {'values': [0.3, 0.5]},
        'activation': {'values': ['relu', 'tanh']},
        'optimizer': {'values': ['adam', 'sgd']},
        'num_filters': {
            'values': [
                [32, 64, 128, 128, 256],
                [16, 32, 64, 64, 128]
            ]
        }
    }
}

def run_sweep():
    sweep_id = wandb.sweep(sweep_config, project="cnn_from_scratch_project")
    wandb.agent(sweep_id, function=launch_training)

def launch_training():
    os.system("python train.py --use_wandb --data_dir data/inaturalist_12K")

if __name__ == "__main__":
    run_sweep()
