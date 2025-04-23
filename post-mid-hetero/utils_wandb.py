# utils_wandb.py
import wandb, os, json, numpy as np

def start_run(args, run_dir):
    """
    Creates/returns a wandb run: the run-name is the same tag used
    for the local folder.  All args are stored in the config.
    """
    wandb.init(
        project="gsat_hetero",
        name=os.path.basename(run_dir),   # tag
        dir=run_dir,                      # wandb stores .wandb file here
        config=vars(args),
        reinit=True
    )
    return wandb.run

def log_epoch(epoch, loss, acc, auc):
    wandb.log({"epoch": epoch, "loss": loss, "acc": acc, "auc": auc})

def log_final(metrics: dict):
    # flatten lists
    flat = {f"{k}_mean": np.mean(v) if isinstance(v, list) else v
            for k, v in metrics.items()}
    wandb.log(flat)
