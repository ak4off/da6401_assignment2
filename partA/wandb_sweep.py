import wandb
import os

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'lr': {'values': [1e-4, 5e-4]},  # Removed >1e-3
        'batch_size': {'values': [32, 64]},
        'dropout': {'values': [0.2, 0.3, 0.4]},  # Focused range
        'activation': {
            'values': ['relu', 'tanh', 'silu', 'mish']  # Removed gelu & leaky_relu
        },
        'optimizer': {'values': ['adam', 'sgd']},
        'num_filters': {
            'values': [
                [32, 64, 128, 128, 256],
                [16, 32, 64, 64, 128]
            ]
        },
        'filter_sizes': {
            'values': [
                [7, 5, 5, 3, 3],
                [3, 3, 5, 5, 7],
                [3, 3, 3, 3, 3],
                [5, 5, 5, 5, 5],
                [7, 7, 7, 7, 7]
            ]
        },
        'dense_neurons': {'values': [256, 512]},
        'batch_norm': {'values': [True]},  # Enforced to help regularization
        'weight_decay': {'values': [1e-4, 1e-3]},  # Removed 0.0
        'use_scheduler': {'values': [True]},  # Enforced ReduceLROnPlateau usage
        'scheduler_patience': {'values': [2, 3, 4]},
        'early_stopping_patience': {'values': [3, 5, 7]},
        'use_data_augmentation': {'values': [True]}  # Enforced
    }
}


def run_sweep():
    sweep_id = wandb.sweep(sweep_config, project="cnn_partA_Assg2")
    wandb.agent(sweep_id, function=launch_training, count=50)  


def launch_training():
    os.system("python train.py --use_wandb --data_dir data/inaturalist_12K")


if __name__ == "__main__":
    run_sweep()