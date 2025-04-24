import wandb, numpy as np, os, pandas as pd

def start_run(args, run_dir):
    return wandb.init(project="gsat_hetero5_seed7_motif1",
                      name=os.path.basename(run_dir),
                      dir=run_dir,
                      config=vars(args), reinit=True)

def log_epoch(epoch, loss, acc, auc):
    wandb.log({"epoch":epoch, "loss":loss, "acc":acc, "auc":auc})

def log_history(loss_h, acc_h, auc_h):
    df = pd.DataFrame({"epoch":range(len(loss_h)),
                       "loss":loss_h, "acc":acc_h, "auc":auc_h})
    wandb.log({"history": wandb.Table(dataframe=df)})

def log_final(metrics:dict):
    flat = {k:(np.mean(v) if isinstance(v,list) else v)
            for k,v in metrics.items()}
    wandb.log(flat)
