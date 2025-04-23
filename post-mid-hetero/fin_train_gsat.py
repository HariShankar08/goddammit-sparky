# train_gsat.py
import argparse, pickle, numpy as np, torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from fin_data import generate_dataset                    # ← the file above
from gsathet2 import convert_to_hetero_data, GSAT_HeteroGNN, GSATTrainer
from fin_eval import evaluate_explanations_multi         # ← new evaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--per_class",type=int,default=50)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--hidden",type=int,default=128)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--out",default="results_seed{}.pkl")
    args=ap.parse_args()

    torch.manual_seed(args.seed);  np.random.seed(args.seed)

    graphs,labels,masks = generate_dataset(args.per_class,args.seed)
    data=[convert_to_hetero_data(g) for g in graphs]
    for d,y in zip(data,labels): d.y=torch.tensor(y)

    train,test=train_test_split(data,test_size=0.2,stratify=labels,random_state=args.seed)
    train_loader=DataLoader(train,batch_size=8,shuffle=True)
    test_loader =DataLoader(test ,batch_size=8)

    rels=[('author','writes','paper'),('paper','cites','paper'),
          ('author','self_loop','author'),('paper','self_loop','paper')]
    model  = GSAT_HeteroGNN(in_dim=66,hidden=args.hidden,out_dim=4,relations=rels,temp=0.5)
    trainer= GSATTrainer(model,lr=args.lr,alpha=.0001,beta=20.)

    for ep in range(args.epochs):
        print(f"Epoch {ep}",trainer.train_epoch(train_loader))
        trainer.test(test_loader)

    metrics=evaluate_explanations_multi(graphs,masks,labels,trainer,convert_to_hetero_data)
    pickle.dump(metrics,open(args.out.format(args.seed),"wb"))
    print("metrics",metrics)

if __name__=="__main__": main()
