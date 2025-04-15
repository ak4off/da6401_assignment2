# # the_sweep.py 

import wandb
from the_trainer import train, parse_args

def sweep_train():
    with wandb.init() as run:
        args = parse_args()

        # Overwrite defaults with current sweep config
        for key, value in run.config.items():
            setattr(args, key, value)

        args.use_wandb = True
        args.wandb_project = run.project
        args.wandb_entity = run.entity
        args.wandb_run_name = run.name

        train(args)

sweep_config = {
    "method": "bayes",  
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    "parameters": {
        "lr": {"min": 1e-5, "max": 1e-3},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "dense_size": {"values": [256, 512, 1024]},
        "freeze_option": {"values": [0, 1, 2]},
        "batch_size": {"values": [32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "weight_decay": {"values": [0.0, 1e-4, 1e-3]},
        "data_aug": {"values": [False, True]},
        # "data_dir": {"values": ['data']}  # Add the data_dir parameter to the sweep
        "data_dir": {"value": "data/inaturalist_12K"},  # Pass dataset path in sweep config
        'use_scheduler': {'values': [True]},  # Enforced ReduceLROnPlateau usage
        'scheduler_patience': {'values': [2, 3, 4]},
        'early_stopping_patience': {'values': [3, 5, 7]},
    }
}

def main():
    sweep_id = wandb.sweep(sweep_config, project="finetune_partB_Assgn2")
    wandb.agent(sweep_id, function=sweep_train, count=30)

if __name__ == "__main__":
    main()