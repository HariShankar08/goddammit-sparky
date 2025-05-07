import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected, to_undirected, k_hop_subgraph
import torch.nn.functional as F
from torch.autograd import Variable
from torch_scatter import scatter
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import copy
import torch_geometric.data.batch as DataBatch

try:
    import higher
except ImportError:
    print("Please install the 'higher' library for meta-learning: pip install higher")
    pass


from pretrain_clf import train_clf_one_seed
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data, relabel
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict


class GSAT(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config, model_config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']
        self.model_name = model_config['model_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.cur_pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']
        self.cur_info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)
        self.sel_r = method_config.get('sel_r', 0.5)

        self.from_scratch = method_config['from_scratch']
        self.save_mcmc = method_config.get('save_mcmc', False)
        self.from_mcmc = method_config.get('from_mcmc', False)
        self.multi_linear = method_config.get('multi_linear', 3)
        self.mcmc_dir = method_config['mcmc_dir']
        self.pre_model_name = method_config['pre_model_name']

        if self.multi_linear in [5552]:
            self.fc_proj = nn.Sequential(nn.Sequential(nn.Dropout(p=0.33),
                                nn.Linear(self.clf.hidden_size, self.clf.hidden_size),
                                nn.BatchNorm1d(self.clf.hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.clf.hidden_size, self.clf.hidden_size),
                            ))
            self.fc_proj = self.fc_proj.to(self.device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            self.optimizer = torch.optim.Adam(list(extractor.parameters()) + list(clf.parameters()) + list(self.fc_proj.parameters()), lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(self.optimizer, mode='max', **scheduler_config)
        elif self.multi_linear in [5553, 5554]:
            class_dim = 1 if num_class == 2 and not multi_label else num_class
            self.fc_proj = nn.Sequential(
                                nn.Sequential(nn.Linear(self.clf.hidden_size, class_dim),
                                nn.BatchNorm1d(class_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(class_dim, class_dim),
                            ))
            self.fc_proj = self.fc_proj.to(self.device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            self.optimizer = torch.optim.Adam(list(extractor.parameters()) + list(clf.parameters()) + list(self.fc_proj.parameters()), lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(self.optimizer, mode='max', **scheduler_config)
        elif self.multi_linear in [5550, 5552, 5553, 5554, 5555, 5559, 5449, 5229, 5669]:
            if not self.from_mcmc:
                self.fc_out = self.clf
            else:
                 self.fc_out = get_model(model_config['x_dim'], model_config['edge_attr_dim'], num_class, model_config['multi_label'], model_config, device)
            lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
            if self.multi_linear in [5552, 5554, 5555, 5669, 5449] and not self.from_mcmc:
                 self.fc_out.load_state_dict(copy.deepcopy(self.clf.state_dict()))
            params_to_optimize = list(extractor.parameters()) + list(self.fc_out.parameters()) if hasattr(self, 'fc_out') else list(extractor.parameters()) + list(clf.parameters())
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
            scheduler_config = method_config.get('scheduler', {})
            self.scheduler = None if scheduler_config == {} else ReduceLROnPlateau(self.optimizer, mode='max', **scheduler_config)

        self.sampling_trials = method_config.get('sampling_trials', 100)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)

    def __loss__(self, att, clf_logits, clf_labels, epoch, training=False, agg='mean'):
        if clf_logits.size(0) != clf_labels.size(0) and clf_labels.size(0) > 0:
            pred_losses = []
            num_trials = clf_logits.size(0) // clf_labels.size(0)
            if num_trials > 1 and clf_logits.size(0) % clf_labels.size(0) == 0:
                 clf_logits_avg = clf_logits.view(num_trials, clf_labels.size(0), -1).mean(dim=0)
                 pred_loss = self.criterion(clf_logits_avg, clf_labels)
                 pred_losses = None
            elif clf_logits.size(0) == self.sampling_trials and clf_labels.size(0) == 1:
                 for i in range(clf_logits.size(0)):
                     pred_losses.append(self.criterion(clf_logits[i, :].unsqueeze(0), clf_labels))
                 if agg.lower() == 'max':
                     pred_loss = torch.stack(pred_losses).max()
                 else:
                     pred_loss = torch.stack(pred_losses).mean()
            else:
                 print(f"Warning: Logits size {clf_logits.size()} mismatch with labels size {clf_labels.size()}. Calculating loss element-wise.")
                 pred_loss = self.criterion(clf_logits, clf_labels)
                 pred_losses = None

        else:
            pred_losses = None
            if clf_labels.size(0) == 0:
                pred_loss = torch.tensor(0.0, device=clf_logits.device, requires_grad=True)
            else:
                pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.final_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        if att.numel() > 0:
             info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()
        else:
             info_loss = torch.tensor(0.0, device=pred_loss.device)

        pred_lossc = pred_loss * self.pred_loss_coef
        info_lossc = info_loss * self.cur_info_loss_coef
        loss = pred_lossc + info_lossc
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        if pred_losses is not None:
            for i, pl in enumerate(pred_losses):
                loss_dict[f'pred_L{i}'] = pl.item()

        if training and self.extractor.parameters():
            pred_grad_list = []
            info_grad_list = []

            self.optimizer.zero_grad()
            if pred_lossc.requires_grad:
                pred_lossc.backward(retain_graph=True)
                for param in self.extractor.parameters():
                    if param.grad is not None:
                        pred_grad_list.append(param.grad.data.clone().flatten().detach())
                pred_grad = torch.cat(pred_grad_list) if pred_grad_list else torch.zeros([1]).to(loss.device)
            else:
                pred_grad = torch.zeros([1]).to(loss.device)


            self.optimizer.zero_grad()
            if info_lossc.requires_grad:
                info_lossc.backward(retain_graph=True)
                for param in self.extractor.parameters():
                    if param.grad is not None:
                        info_grad_list.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                info_grad = torch.cat(info_grad_list) if info_grad_list else torch.zeros([1]).to(loss.device)
            else:
                info_grad = torch.zeros([1]).to(loss.device)


            if pred_grad.numel() > 1 and info_grad.numel() > 1:
                 grad_sim = F.cosine_similarity(pred_grad.unsqueeze(0), info_grad.unsqueeze(0)).to(loss.device)
                 loss_dict['grad_sim'] = grad_sim.item()
            else:
                 loss_dict['grad_sim'] = 0.0

            loss_dict['pred_grad'] = pred_grad.norm().item()
            loss_dict['info_grad'] = info_grad.norm().item()

            self.optimizer.zero_grad()

        return loss, loss_dict


    def package_subgraph(self, data, att_bern, epoch, verbose=False):
        b = torch.bernoulli(att_bern)
        att_binary = (b - att_bern).detach() + att_bern

        def relabel(x, edge_index, batch, pos=None):
            num_nodes = x.size(0)
            sub_nodes = torch.unique(edge_index)
            if sub_nodes.numel() == 0:
                return x.new_empty((0, x.size(1))), edge_index.new_empty((2,0)), batch.new_empty((0,)), None if pos is None else pos.new_empty((0, pos.size(1)))

            x = x[sub_nodes]
            batch = batch[sub_nodes]
            row, col = edge_index
            node_idx = row.new_full((num_nodes,), -1)
            node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
            edge_index = node_idx[edge_index]
            if pos is not None:
                pos = pos[sub_nodes]
            return x, edge_index, batch, pos

        idx_reserve = torch.nonzero(att_binary == 1, as_tuple=True)[0]
        if verbose:
            print(len(idx_reserve) / len(att_binary) if len(att_binary) > 0 else 0, self.get_r(
                self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r))

        if idx_reserve.numel() == 0:
             from torch_geometric.data import Data
             return Data(x=data.x.new_empty((0, data.x.size(1))),
                         edge_index=data.edge_index.new_empty((2,0)),
                         batch=data.batch.new_empty((0,)),
                         edge_attr=data.edge_attr.new_empty((0, data.edge_attr.size(1))) if data.edge_attr is not None else None,
                         edge_atten=att_binary.new_empty((0,)),
                         graph_prob=0)


        causal_edge_index = data.edge_index[:, idx_reserve]
        if data.edge_attr is not None:
            causal_edge_attr = data.edge_attr[idx_reserve]
        else:
            causal_edge_attr = None
        causal_edge_atten = att_binary[idx_reserve]

        nodes_in_causal_graph = torch.unique(causal_edge_index)
        if nodes_in_causal_graph.numel() == 0:
             from torch_geometric.data import Data
             return Data(x=data.x.new_empty((0, data.x.size(1))),
                         edge_index=data.edge_index.new_empty((2,0)),
                         batch=data.batch.new_empty((0,)),
                         edge_attr=data.edge_attr.new_empty((0, data.edge_attr.size(1))) if data.edge_attr is not None else None,
                         edge_atten=att_binary.new_empty((0,)),
                         graph_prob=0)

        causal_x = data.x[nodes_in_causal_graph]
        causal_batch = data.batch[nodes_in_causal_graph]
        node_map = nodes_in_causal_graph.new_full((data.x.size(0),), -1)
        node_map[nodes_in_causal_graph] = torch.arange(nodes_in_causal_graph.size(0), device=data.x.device)
        causal_edge_index = node_map[causal_edge_index]


        graph_prob = 0
        from torch_geometric.data import Data
        subgraph_data = Data(x=causal_x, edge_index=causal_edge_index, batch=causal_batch,
                             edge_attr=causal_edge_attr, edge_atten=causal_edge_atten, graph_prob=graph_prob, y=data.y)
        return subgraph_data


    def attend(self, data, att_log_logits, epoch, training):
        att = self.sampling(att_log_logits, temp=1, training=training)
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                if att.numel() == data.edge_index.size(1):
                    trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                    trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                    edge_att = (att + trans_val_perm) / 2
                else:
                    print(f"Warning: Attention size {att.shape} mismatch with edge_index size {data.edge_index.shape}. Using raw attention.")
                    edge_att = att
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
        return edge_att

    def split_graph(self, data, edge_score, ratio):
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)
            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            if index.numel() == 0:
                return index.new_empty((0,), dtype=torch.long), index.new_empty((0,), dtype=torch.long), src.new_empty((0,)), index.new_empty((0,), dtype=torch.long), index.new_empty((0,), dtype=torch.bool)

            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            graph_indices = data.batch[data.edge_index[0]]
            num_graphs = data.batch.max().item() + 1
            num_edges_per_graph = scatter(torch.ones_like(graph_indices), graph_indices, dim=0, dim_size=num_graphs, reduce='sum')


            k = (ratio * num_edges_per_graph.to(float)).ceil().to(torch.long)

            start_indices = torch.cat([torch.zeros((1,), device=src.device, dtype=torch.long), num_edges_per_graph.cumsum(0)])

            mask = []
            current_start = 0
            for i in range(num_graphs):
                 graph_k = k[i]
                 graph_num_edges = num_edges_per_graph[i]
                 actual_k = min(graph_k, graph_num_edges)
                 if actual_k > 0:
                     mask.append(torch.arange(actual_k, dtype=torch.long, device=src.device) + start_indices[i])
                 current_start += graph_num_edges

            if not mask:
                 final_mask_indices = torch.tensor([], dtype=torch.long, device=src.device)
            else:
                 final_mask_indices = torch.cat(mask, dim=0)


            bool_mask = torch.zeros_like(index, device=index.device, dtype=torch.bool)
            if final_mask_indices.numel() > 0:
                 topk_indices_in_perm = final_mask_indices
                 bool_mask.scatter_(0, perm[topk_indices_in_perm], True)

            topk_perm = perm[bool_mask[perm]]
            exc_perm = perm[~bool_mask[perm]]

            return topk_perm, exc_perm, rank, perm, bool_mask


        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        if data.edge_index.numel() == 0 or edge_score.numel() == 0:
             return data.edge_index.new_empty((0,), dtype=torch.long), data.edge_index.new_empty((0,), dtype=torch.long)

        graph_indices = data.batch[data.edge_index[0]]
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score.view(-1), graph_indices, ratio, descending=True)

        return new_idx_reserve, new_idx_drop


    def forward_pass(self, data, epoch, training):
        if data.x is None or data.x.size(0) == 0 or data.edge_index is None:
             print("Warning: forward_pass received empty graph data.")
             dummy_att = torch.tensor([], device=self.device)
             dummy_logits = torch.tensor([], device=self.device)
             loss = torch.tensor(0.0, device=self.device, requires_grad=training)
             loss_dict = {'loss': 0.0, 'pred': 0.0, 'info': 0.0}
             return dummy_att, loss, loss_dict, dummy_logits


        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)

        if self.multi_linear == 3:
            edge_att = self.attend(data, att_log_logits, epoch, training)
            clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att, att_opt='first')
            att_sigmoid = att_log_logits.sigmoid() if att_log_logits.numel() > 0 else torch.tensor([], device=att_log_logits.device)
            loss, loss_dict = self.__loss__(att_sigmoid, clf_logits, data.y, epoch, training)

        elif self.multi_linear == 8:
            sampling_logits = []
            cur_r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if self.decay_interval is not None and (cur_r == 1 or epoch < self.decay_interval):
                 num_trials = 1
                 self.cur_info_loss_coef = 2 * self.info_loss_coef
            else:
                 num_trials = self.sampling_trials
                 self.cur_info_loss_coef = self.info_loss_coef

            for _ in range(num_trials):
                edge_att_cont = self.attend(data, att_log_logits, epoch, training)
                b = torch.bernoulli(edge_att_cont)
                cur_edge_att_binary_ste = (b - edge_att_cont).detach() + edge_att_cont
                clf_logits_trial = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=cur_edge_att_binary_ste)
                sampling_logits.append(clf_logits_trial)

            clf_logits = torch.stack(sampling_logits).mean(dim=0)
            att_sigmoid = att_log_logits.sigmoid() if att_log_logits.numel() > 0 else torch.tensor([], device=att_log_logits.device)
            loss, loss_dict = self.__loss__(att_sigmoid, clf_logits, data.y, epoch, training)
            self.cur_info_loss_coef = self.info_loss_coef

        elif self.multi_linear in [5550, 5552, 5553, 5554, 5555, 5559, 5449, 5229, 5669]:
             att_log_logits = att_log_logits.detach()
             edge_att = self.attend(data, att_log_logits, epoch, training=False)

             if self.multi_linear in [5550, 5552, 5553, 5554, 5555]:
                 new_idx_reserve, new_idx_drop = self.split_graph(data, edge_att, self.sel_r)
                 causal_edge_weight = edge_att.clone()
                 if self.multi_linear in [5553, 5554, 5555]:
                     causal_edge_weight[new_idx_reserve] = 1
                 if self.multi_linear in [5550, 5552, 5555]:
                     causal_edge_weight[new_idx_drop] = 0
                 clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=causal_edge_weight)

             elif self.multi_linear in [5559, 5669]:
                 clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)

             elif self.multi_linear in [5449, 5229]:
                 clf_logits = self.fc_out(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att, att_opt='first')
             else:
                 raise ValueError(f"Unhandled multi_linear value in specific range: {self.multi_linear}")

             loss = self.criterion(clf_logits, data.y)
             att_sigmoid = att_log_logits.sigmoid() if att_log_logits.numel() > 0 else torch.tensor([], device=att_log_logits.device)
             _, loss_dict = self.__loss__(att_sigmoid, clf_logits, data.y, epoch, training=False)
             loss_dict['loss'] = loss.item()


        else:
            print(f"Warning: multi_linear={self.multi_linear} not explicitly handled. Using default behavior (like GMT-LIN without 'first' option).")
            edge_att = self.attend(data, att_log_logits, epoch, training)
            clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
            att_sigmoid = att_log_logits.sigmoid() if att_log_logits.numel() > 0 else torch.tensor([], device=att_log_logits.device)
            loss, loss_dict = self.__loss__(att_sigmoid, clf_logits, data.y, epoch, training)

        edge_att_eval = att_log_logits.sigmoid().detach()
        if edge_att_eval.numel() > 0:
            if self.learn_edge_att:
                if is_undirected(data.edge_index):
                    if edge_att_eval.numel() == data.edge_index.size(1):
                        trans_idx, trans_val = transpose(data.edge_index, edge_att_eval, None, None, coalesced=False)
                        trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                        edge_att_eval = (edge_att_eval + trans_val_perm) / 2

            else:
                edge_att_eval = self.lift_node_att_to_edge_att(edge_att_eval, data.edge_index)


        return edge_att_eval, loss, loss_dict, clf_logits


    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        self.eval()
        if data.x is None or data.x.size(0) == 0:
             print("Warning: eval_one_batch received empty data.")
             return torch.tensor([]), {'loss': 0.0, 'pred': 0.0, 'info': 0.0}, torch.tensor([])

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.cpu().reshape(-1), loss_dict, clf_logits.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()
        self.train()
        if data.x is None or data.x.size(0) == 0:
             print("Warning: train_one_batch received empty data.")
             return torch.tensor([]), {'loss': 0.0, 'pred': 0.0, 'info': 0.0}, torch.tensor([])

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)

        if loss.requires_grad:
             self.optimizer.zero_grad()
             loss.backward()
             self.optimizer.step()


        return att.cpu().reshape(-1), loss_dict, clf_logits.cpu()

    def extract_computational_subgraph(self, data, v, hops=1):
        """
        Extract the k-hop subgraph around node v from the batch data.
        Handles potential edge cases like isolated nodes or empty subgraphs.
        """
        if v >= data.num_nodes:
             print(f"Warning: Node index {v} out of bounds for graph with {data.num_nodes} nodes.")
             from torch_geometric.data import Data
             return Data(x=data.x.new_empty((0, data.x.size(1))),
                         edge_index=data.edge_index.new_empty((2,0)),
                         batch=data.batch.new_empty((0,)),
                         y=data.y.new_empty((0,) if data.y.dim() > 1 else ()) )


        node_idx, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            v, hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
        )

        if node_idx.numel() == 0:
             from torch_geometric.data import Data
             return Data(x=data.x.new_empty((0, data.x.size(1))),
                         edge_index=data.edge_index.new_empty((2,0)),
                         batch=data.batch.new_empty((0,)),
                         y=data.y.new_empty((0,) if data.y.dim() > 1 else ()) )


        sub_x = data.x[node_idx]
        sub_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None and edge_mask.numel() > 0 else None
        sub_batch = data.batch[node_idx]

        graph_idx = data.batch[v]
        if data.y.size(0) == data.num_graphs:
             sub_y = data.y[graph_idx].unsqueeze(0)
        elif data.y.size(0) == data.num_nodes:
             sub_y = data.y[node_idx]
        else:
             print(f"Warning: Unexpected label shape {data.y.shape}. Using full label tensor for subgraph.")
             sub_y = data.y

        from torch_geometric.data import Data
        sub_data = Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr, batch=sub_batch, y=sub_y)
        sub_data.num_graphs = 1
        return sub_data


    def explanation_loss(self, sub_data, fmodel, epoch):
        """
        Compute an explanation loss on the computational subgraph using the adapted model (fmodel).
        Uses info-loss based on the extractor's output for the subgraph.
        """
        if sub_data.x is None or sub_data.x.size(0) == 0 or sub_data.edge_index is None:
             return torch.tensor(0.0, device=self.device, requires_grad=True)

        emb = fmodel.get_emb(sub_data.x, sub_data.edge_index, batch=sub_data.batch, edge_attr=sub_data.edge_attr)
        att_log_logits = self.extractor(emb, sub_data.edge_index, sub_data.batch)

        if att_log_logits.numel() == 0:
             return torch.tensor(0.0, device=self.device, requires_grad=True)

        att = att_log_logits.sigmoid()
        r_target = self.final_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)

        info_loss = (att * torch.log(att / r_target + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r_target + 1e-6) + 1e-6)).mean()
        return info_loss


    def classification_loss(self, data, fmodel):
        """
        Compute classification loss using the adapted model fmodel on the full data.
        This uses the standard forward pass of the adapted classifier.
        It does NOT involve the SAMPLING mechanism, even if multi_linear is 8.
        The meta-update aims to make the base classifier perform well after being
        adapted based on subgraph explanations.
        """
        if data.x is None or data.x.size(0) == 0:
             return torch.tensor(0.0, device=self.device, requires_grad=True)

        clf_logits = fmodel(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)

        if data.y is None or data.y.numel() == 0:
             return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = self.criterion(clf_logits, data.y)
        return loss

    def train_self_meta(self, loaders, test_set, metric_dict, use_edge_attr, inner_steps=3, adapt_steps=3, alpha=0.01, beta=0.01, meta_batch_size=1, hops=1):
        """
        Meta-learning training loop using 'higher'.
        Adapts the classifier (clf) based on explanation loss on subgraphs,
        then updates original clf and extractor based on classification loss on full graphs.
        """
        if 'higher' not in sys.modules:
             raise ImportError("Meta-learning requires the 'higher' library. Please install it.")

        train_loader = loaders['train']

        for epoch in range(self.epochs):
            self.clf.train()
            self.extractor.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Meta]")

            for idx, data in enumerate(pbar):
                data = process_data(data, use_edge_attr)
                data = data.to(self.device)

                if data.x is None or data.x.size(0) == 0 or data.num_nodes == 0:
                    print(f"Skipping empty batch {idx}")
                    continue

                self.optimizer.zero_grad()

                if data.num_nodes <= 0:
                     print(f"Skipping batch {idx} due to zero nodes.")
                     continue
                v = np.random.choice(data.num_nodes)
                sub_data = self.extract_computational_subgraph(data, v, hops=hops)

                if sub_data.x is None or sub_data.x.size(0) == 0:
                     print(f"Skipping batch {idx}, node {v}: Empty subgraph.")
                     continue


                inner_opt = torch.optim.SGD(self.clf.parameters(), lr=alpha)
                with higher.innerloop_ctx(self.clf, inner_opt, copy_initial_weights=True, track_higher_grads=True) as (fmodel, diffopt):
                    for _ in range(adapt_steps):
                        expl_loss = self.explanation_loss(sub_data, fmodel, epoch)

                        if expl_loss.requires_grad:
                             diffopt.step(expl_loss)


                    outer_loss = self.classification_loss(data, fmodel)

                    if outer_loss.requires_grad:
                         outer_loss.backward()


                self.optimizer.step()

                pbar.set_postfix(OuterLoss=outer_loss.item())


            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'val', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('meta_train/lr', get_lr(self.optimizer), epoch)

            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2
            if self.scheduler is not None:
                if valid_res and len(valid_res) > main_metric_idx and valid_res[main_metric_idx] is not None:
                     self.scheduler.step(valid_res[main_metric_idx])


            r = self.final_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if valid_res and len(valid_res) > max(main_metric_idx, 4) and valid_res[main_metric_idx] is not None and valid_res[4] is not None:
                 if (r == self.final_r or self.fix_r) and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                            or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):
                     metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                                    'metric/best_clf_train': train_res[main_metric_idx] if train_res else None,
                                    'metric/best_clf_valid': valid_res[main_metric_idx],
                                    'metric/best_clf_test': test_res[main_metric_idx] if test_res else None,
                                    'metric/best_x_roc_train': train_res[0] if train_res else None,
                                    'metric/best_x_roc_valid': valid_res[0],
                                    'metric/best_x_roc_test': test_res[0] if test_res else None,
                                    'metric/best_x_precision_train': train_res[1] if train_res else None,
                                    'metric/best_x_precision_valid': valid_res[1],
                                    'metric/best_x_precision_test': test_res[1] if test_res else None,
                                    'metric/best_tvd_train': train_res[5] if train_res and len(train_res)>5 else None,
                                    'metric/best_tvd_valid': valid_res[5] if valid_res and len(valid_res)>5 else None,
                                    'metric/best_tvd_test': test_res[5] if test_res and len(test_res)>5 else None}
                     if self.save_mcmc:
                         save_checkpoint(self.clf, self.mcmc_dir, model_name=self.pre_model_name + f"_clf_mcmc")
                         save_checkpoint(self.extractor, self.mcmc_dir, model_name=self.pre_model_name + f"_att_mcmc")


            for metric, value in metric_dict.items():
                if value is not None:
                     metric_short = metric.split('/')[-1]
                     self.writer.add_scalar(f'xgnn_best/{metric_short}', value, epoch)

            print(f'[Meta Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}')
            print('====================================')
        return metric_dict

    def train_self(self, loaders, test_set, metric_dict, use_edge_attr):
        """ Standard training loop (without meta-learning) """
        viz_set = []
        if self.num_viz_samples > 0 and test_set is not None:
             try:
                 viz_set = self.get_viz_idx(test_set, self.dataset_name)
             except Exception as e:
                 print(f"Warning: Could not get visualization indices: {e}")


        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'val', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('xgnn_train/lr', get_lr(self.optimizer), epoch)

            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2
            if self.scheduler is not None:
                 if valid_res and len(valid_res) > main_metric_idx and valid_res[main_metric_idx] is not None:
                     self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if valid_res and len(valid_res) > max(main_metric_idx, 4) and valid_res[main_metric_idx] is not None and valid_res[4] is not None:
                 if (r == self.final_r or self.fix_r is not None) and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                            or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):
                     metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                                    'metric/best_clf_train': train_res[main_metric_idx] if train_res else None,
                                    'metric/best_clf_valid': valid_res[main_metric_idx],
                                    'metric/best_clf_test': test_res[main_metric_idx] if test_res else None,
                                    'metric/best_x_roc_train': train_res[0] if train_res else None,
                                    'metric/best_x_roc_valid': valid_res[0],
                                    'metric/best_x_roc_test': test_res[0] if test_res else None,
                                    'metric/best_x_precision_train': train_res[1] if train_res else None,
                                    'metric/best_x_precision_valid': valid_res[1],
                                    'metric/best_x_precision_test': test_res[1] if test_res else None,
                                    'metric/best_tvd_train': train_res[5] if train_res and len(train_res)>5 else None,
                                    'metric/best_tvd_valid': valid_res[5] if valid_res and len(valid_res)>5 else None,
                                    'metric/best_tvd_test': test_res[5] if test_res and len(test_res)>5 else None}
                     if self.save_mcmc:
                         if self.mcmc_dir and self.pre_model_name:
                             try:
                                 save_checkpoint(self.clf, self.mcmc_dir, model_name=self.pre_model_name + f"_clf_mcmc")
                                 save_checkpoint(self.extractor, self.mcmc_dir, model_name=self.pre_model_name + f"_att_mcmc")
                             except Exception as e:
                                 print(f"Error saving MCMC checkpoint: {e}")

            for metric, value in metric_dict.items():
                 if value is not None:
                     metric_short = metric.split('/')[-1]
                     self.writer.add_scalar(f'xgnn_best/{metric_short}', value, epoch)

            if self.num_viz_samples != 0 and viz_set and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    print("Warning: Visualization for multi-label not implemented.")
                elif test_set is not None:
                    for idx_list, tag in viz_set:
                        try:
                            self.visualize_results(test_set, idx_list, epoch, tag, use_edge_attr)
                        except Exception as e:
                            print(f"Error during visualization for tag {tag}: {e}")


            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}')
            print('====================================')
        return metric_dict

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase_label = 'test ' if phase == 'test' else phase

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for _ in range(5))
        pbar = tqdm(data_loader, desc=f"Epoch {epoch} [{phase_label.strip()}]")

        for idx, data in enumerate(pbar):
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'batch') or not hasattr(data, 'y'):
                 print(f"Warning: Skipping invalid data batch {idx} in {phase} phase.")
                 continue

            data = process_data(data, use_edge_attr)
            if data.x is None or data.x.size(0) == 0:
                 print(f"Warning: Skipping empty batch {idx} after processing in {phase} phase.")
                 continue

            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            if att is None or loss_dict is None or clf_logits is None:
                 print(f"Warning: Skipping batch {idx} due to None result from run_one_batch in {phase} phase.")
                 continue


            exp_labels = getattr(data, 'edge_label', None)
            if exp_labels is not None and isinstance(exp_labels, torch.Tensor):
                 exp_labels_cpu = exp_labels.data.cpu()
                 if att.numel() > 0 and exp_labels_cpu.numel() > 0:
                     try:
                         precision_at_k = self.get_precision_at_k(att, exp_labels_cpu, self.k, data.batch, data.edge_index)
                         all_precision_at_k.extend(precision_at_k)
                     except Exception as e:
                         print(f"Error calculating precision@k for batch {idx}: {e}")
                 else:
                     precision_at_k = []

                 all_exp_labels.append(exp_labels_cpu)
                 all_att.append(att)
            else:
                 precision_at_k = []


            clf_labels_cpu = data.y.data.cpu()
            all_clf_labels.append(clf_labels_cpu)
            all_clf_logits.append(clf_logits)

            if clf_labels_cpu.numel() > 0 and clf_logits.numel() > 0:
                 try:
                     desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels_cpu if exp_labels is not None else None, att, precision_at_k,
                                                           clf_labels_cpu, clf_logits, batch=True)
                     pbar.set_description(desc)
                 except Exception as e:
                     print(f"Error logging batch {idx} metrics: {e}")

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v


        if not all_att:
             print(f"Warning: No valid batches processed in epoch {epoch}, phase {phase}.")
             return None, None, None, None, None, None


        try:
            all_exp_labels_cat = torch.cat(all_exp_labels) if all_exp_labels else torch.tensor([])
            all_att_cat = torch.cat(all_att) if all_att else torch.tensor([])
            all_clf_labels_cat = torch.cat(all_clf_labels) if all_clf_labels else torch.tensor([])
            all_clf_logits_cat = torch.cat(all_clf_logits) if all_clf_logits else torch.tensor([])
        except Exception as e:
            print(f"Error concatenating results for epoch {epoch}, phase {phase}: {e}")
            return None, None, None, None, None, None


        final_loss_dict = {}
        if loader_len > 0:
            for k, v in all_loss_dict.items():
                final_loss_dict[k] = v / loader_len
        else:
            final_loss_dict = all_loss_dict


        try:
            desc, att_auroc, precision, clf_acc, clf_roc, pred_loss = self.log_epoch(
                epoch, phase, final_loss_dict,
                all_exp_labels_cat, all_att_cat, all_precision_at_k,
                all_clf_labels_cat, all_clf_logits_cat, batch=False
            )
            pbar.set_description(desc)
        except Exception as e:
            print(f"Error logging epoch {epoch} summary metrics: {e}")
            att_auroc, precision, clf_acc, clf_roc, pred_loss = None, None, None, None, None

        tvd_metric = final_loss_dict.get('logits_tvd', -1)
        return att_auroc, precision, clf_acc, clf_roc, pred_loss, tvd_metric


    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        phase_str = phase.replace("_", " ")
        desc = f'[Seed {self.random_state}, Epoch {epoch}]: xgnn_{phase_str}........., ' if batch else f'[Seed {self.random_state}, Epoch {epoch}]: xgnn_{phase_str} finished, '
        for k, v in loss_dict.items():
            if not batch and v is not None:
                self.writer.add_scalar(f'xgnn_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, ' if v is not None else f'{k}: N/A, '

        att_auroc, precision, clf_acc, clf_roc = None, None, None, None
        eval_desc = ""
        if not batch and exp_labels is not None and att is not None and clf_labels is not None and clf_logits is not None and \
           exp_labels.numel() > 0 and att.numel() > 0 and clf_labels.numel() > 0 and clf_logits.numel() > 0:
             try:
                 eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
             except Exception as e:
                 print(f"Error calculating eval scores for epoch {epoch}, phase {phase}: {e}")
                 eval_desc = "Eval Error"

        desc += eval_desc

        tvd_val = loss_dict.get('logits_tvd', None)
        if tvd_val is not None and not batch:
             if att_auroc is not None and clf_acc is not None:
                 self.writer.add_scalar(f'xgnn_{phase}/tvd_gap_acc', att_auroc - clf_acc, epoch)
             if att_auroc is not None and clf_roc is not None:
                 self.writer.add_scalar(f'xgnn_{phase}/tvd_gap_roc', att_auroc - clf_roc, epoch)

        pred_loss_val = loss_dict.get('pred', None)
        return desc, att_auroc, precision, clf_acc, clf_roc, pred_loss_val


    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        if not all(isinstance(t, torch.Tensor) for t in [exp_labels, att, clf_labels, clf_logits]):
             print("Warning: Invalid input types for get_eval_score.")
             return "Invalid Input", None, None, None, None

        if exp_labels.numel() == 0 or att.numel() == 0 or clf_labels.numel() == 0 or clf_logits.numel() == 0:
             print("Warning: Empty tensors provided to get_eval_score.")
             clf_acc = 0.0
             if batch:
                 return f'clf_acc: {clf_acc:.3f}', None, None, None, None
             else:
                 return f'clf_acc: {clf_acc:.3f}, clf_roc: N/A, att_roc: N/A, att_prec@k: N/A', 0.0, 0.0, 0.0, 0.0


        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0.0
        if clf_labels.shape == clf_preds.shape:
             clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]


        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k_mean = np.mean(precision_at_k) if precision_at_k else 0.0

        clf_roc = 0.0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            try:
                 y_true_np = clf_labels.view(-1, 1).numpy() if clf_labels.dim() == 1 else clf_labels.numpy()
                 y_pred_np = clf_logits.numpy()
                 input_dict = {'y_true': y_true_np, 'y_pred': y_pred_np}
                 clf_roc = evaluator.eval(input_dict)['rocauc']
            except Exception as e:
                 print(f"Error using OGB evaluator: {e}")
                 clf_roc = 0.0

        att_auroc = 0.0
        bkg_att_weights = torch.tensor([])
        signal_att_weights = torch.tensor([])
        if exp_labels.numel() > 0 and torch.unique(exp_labels).shape[0] > 1:
             try:
                 if exp_labels.shape == att.shape:
                     att_auroc = roc_auc_score(exp_labels.numpy(), att.numpy())
                     bkg_att_weights = att[exp_labels == 0]
                     signal_att_weights = att[exp_labels == 1]
             except Exception as e:
                 print(f"Error calculating attention AUROC: {e}")
                 att_auroc = 0.0


        if bkg_att_weights.numel() > 0:
             self.writer.add_histogram(f'xgnn_{phase}/bkg_att_weights', bkg_att_weights, epoch)
             self.writer.add_scalar(f'xgnn_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean().item(), epoch)
        if signal_att_weights.numel() > 0:
             self.writer.add_histogram(f'xgnn_{phase}/signal_att_weights', signal_att_weights, epoch)
             self.writer.add_scalar(f'xgnn_{phase}/avg_signal_att_weights/', signal_att_weights.mean().item(), epoch)

        self.writer.add_scalar(f'xgnn_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'xgnn_{phase}/precision@{self.k}/', precision_at_k_mean, epoch)
        if att.numel() > 0:
             self.writer.add_scalar(f'xgnn_{phase}/avg_att_weights_std/', att.std().item(), epoch)

        if exp_labels.numel() > 0 and torch.unique(exp_labels).shape[0] > 1 and exp_labels.shape == att.shape:
             try:
                 self.writer.add_pr_curve(f'PR_Curve/xgnn_{phase}/', exp_labels, att, epoch)
             except Exception as e:
                 print(f"Error adding PR curve: {e}")


        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k_mean:.3f}'
        return desc, att_auroc, precision_at_k_mean, clf_acc, clf_roc


    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        if not all(isinstance(t, torch.Tensor) for t in [att, exp_labels, batch, edge_index]):
             print("Warning: Invalid input types for get_precision_at_k.")
             return []
        if edge_index.size(1) != att.numel() or edge_index.size(1) != exp_labels.numel():
             print(f"Warning: Size mismatch in get_precision_at_k. Edges: {edge_index.size(1)}, Att: {att.numel()}, Labels: {exp_labels.numel()}")
             return []


        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0

        for i in range(num_graphs):
            nodes_for_graph_i = torch.nonzero(batch == i, as_tuple=False).view(-1)
            if nodes_for_graph_i.numel() == 0: continue

            edge_mask_graph_i = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)

            if not torch.any(edge_mask_graph_i): continue

            labels_for_graph_i = exp_labels[edge_mask_graph_i]
            att_for_graph_i = att[edge_mask_graph_i]

            actual_k = min(k, att_for_graph_i.numel())
            if actual_k <= 0:
                 precision_at_k.append(0.0)
                 continue

            _, top_k_indices = torch.topk(att_for_graph_i, actual_k)

            precision = labels_for_graph_i[top_k_indices].sum().item() / actual_k
            precision_at_k.append(precision)

        return precision_at_k


    def get_viz_idx(self, test_set, dataset_name):
        if test_set is None or len(test_set) == 0:
             print("Warning: Test set is empty or None, cannot select visualization indices.")
             return []

        if not hasattr(test_set, 'data') or not hasattr(test_set.data, 'y'):
             print("Warning: Test set data missing 'y' attribute.")
             return []

        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = []
        for i in range(len(test_set)):
             try:
                 num_nodes.append(test_set[i].x.shape[0])
             except AttributeError:
                 print(f"Warning: Could not get node count for test sample {i}.")
                 num_nodes.append(0)
        num_nodes = np.array(num_nodes)


        classes = np.unique(y_dist)
        res = []
        if self.num_viz_samples <= 0: return []

        for each_class in classes:
            tag = 'class_' + str(each_class)
            candidate_indices = np.where(y_dist == each_class)[0]

            if dataset_name == 'Graph-SST2':
                 size_cond = (num_nodes > 5) & (num_nodes < 10)
                 valid_indices = candidate_indices[size_cond[candidate_indices]]
            else:
                 valid_indices = candidate_indices

            if len(valid_indices) == 0:
                 print(f"Warning: No valid samples found for class {each_class} to visualize.")
                 continue

            num_to_sample = min(self.num_viz_samples, len(valid_indices))
            replace = num_to_sample > len(valid_indices)
            try:
                 chosen_idx = np.random.choice(valid_indices, num_to_sample, replace=replace)
                 res.append((chosen_idx, tag))
            except Exception as e:
                 print(f"Error choosing visualization samples for class {each_class}: {e}")


        return res


    def visualize_results(self, test_set, idx_list, epoch, tag, use_edge_attr):
        if test_set is None or len(test_set) == 0 or len(idx_list) == 0:
             print("Skipping visualization due to empty test set or index list.")
             return

        try:
             viz_set = test_set[idx_list]
        except IndexError:
             print(f"Error: Indices out of bounds for visualization. Max index: {len(test_set)-1}, Requested: {idx_list}")
             return
        except Exception as e:
             print(f"Error creating visualization subset: {e}")
             return


        viz_loader = DataLoader(viz_set, batch_size=len(idx_list), shuffle=False)
        try:
             data = next(iter(viz_loader))
        except StopIteration:
             print("Error: Visualization loader is empty.")
             return

        data = process_data(data, use_edge_attr)
        data = data.to(self.device)

        batch_att, _, _ = self.eval_one_batch(data, epoch)
        batch_att = batch_att.cpu()

        imgs = []
        current_edge_idx = 0
        for i in range(len(viz_set)):
            sample_data = viz_set[i]
            mol_type, coor = None, None

            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                node_types = getattr(sample_data, 'node_type', None)
                if node_types is not None:
                     mol_type = {k: node_dict.get(v.item(), '?') for k, v in enumerate(node_types)}
                else: mol_type = {k: '?' for k in range(sample_data.num_nodes)}

            elif self.dataset_name == 'Graph-SST2':
                 tokens = getattr(sample_data, 'sentence_tokens', None)
                 if tokens:
                     mol_type = {k: v for k, v in enumerate(tokens)}
                 else: mol_type = {k: '?' for k in range(sample_data.num_nodes)}
                 num_nodes_sample = sample_data.num_nodes
                 if num_nodes_sample > 0:
                     x_coords = np.linspace(0, 1, num_nodes_sample)
                     y_coords = np.ones_like(x_coords)
                     coor = np.stack([x_coords, y_coords], axis=1)

            elif self.dataset_name == 'ogbg_molhiv':
                 if hasattr(sample_data, 'x') and sample_data.x.size(1) > 0:
                     try:
                         element_idxs = {k: int(v.item() + 1) for k, v in enumerate(sample_data.x[:, 0])}
                         mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), v) for k, v in element_idxs.items()}
                     except Exception as e:
                         print(f"Error getting element symbols for ogbg_molhiv sample {i}: {e}")
                         mol_type = {k: '?' for k in range(sample_data.num_nodes)}
                 else: mol_type = {k: '?' for k in range(sample_data.num_nodes)}

            elif self.dataset_name == 'mnist':
                 print("Visualization for mnist not implemented.")
                 continue
            else:
                 mol_type = {k: str(k) for k in range(sample_data.num_nodes)}


            num_edges_sample = sample_data.edge_index.size(1)
            if current_edge_idx + num_edges_sample > batch_att.size(0):
                 print(f"Warning: Edge index mismatch during visualization for sample {i}. Skipping.")
                 continue

            edge_att_sample = batch_att[current_edge_idx : current_edge_idx + num_edges_sample]
            current_edge_idx += num_edges_sample

            node_label = getattr(sample_data, 'node_label', None)
            if node_label is None:
                 node_label = torch.zeros(sample_data.num_nodes)
            node_label = node_label.reshape(-1)


            try:
                 fig, img = visualize_a_graph(
                     sample_data.edge_index.cpu(),
                     edge_att_sample.cpu(),
                     node_label.cpu(),
                     self.dataset_name,
                     norm=self.viz_norm_att,
                     mol_type=mol_type,
                     coor=coor
                 )
                 imgs.append(img)
                 plt.close(fig)
            except Exception as e:
                 print(f"Error visualizing graph {i} for tag {tag}: {e}")

        if imgs:
            try:
                imgs_np = np.stack(imgs)
                self.writer.add_images(tag, imgs_np, epoch, dataformats='NHWC')
            except Exception as e:
                print(f"Error stacking images or adding to TensorBoard: {e}")


    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        if decay_interval is None or decay_r is None or decay_interval <= 0:
             return final_r

        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        if r > init_r:
             r = init_r
        return r


    def sampling(self, att_log_logits, temp, training):
        if att_log_logits.numel() == 0:
             return torch.tensor([], device=att_log_logits.device)
        att = self.concrete_sample(att_log_logits, temp=temp, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        if node_att.numel() == 0 or edge_index.numel() == 0:
             return torch.tensor([], device=node_att.device)

        if edge_index.max() >= node_att.size(0):
             print(f"Warning: Invalid edge index {edge_index.max()} for node attention size {node_att.size(0)}")
             return torch.tensor([], device=node_att.device)


        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def lift_edge_att_to_node_att(edge_att, edge_index, size=None):
        if edge_att.numel() == 0 or edge_index.numel() == 0:
             return torch.tensor([], device=edge_att.device) if size is None else torch.ones(size, device=edge_att.device)

        if size is None:
             size = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0

        if size == 0:
             return torch.tensor([], device=edge_att.device)


        if edge_index.size(1) != edge_att.size(0):
             print(f"Warning: Mismatch edge_att size {edge_att.size(0)} and edge_index size {edge_index.size(1)}")
             return torch.ones(size, device=edge_att.device)


        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        if src_idx.max() >= size:
             print(f"Warning: Invalid source index {src_idx.max()} for size {size}")
             return torch.ones(size, device=edge_att.device)


        src_scatter = scatter(edge_att, src_idx, reduce='mul', dim=0, dim_size=size)
        dst_scatter = scatter(edge_att, edge_index[1], reduce='mul', dim=0, dim_size=size)

        node_att_aggregated = src_scatter * dst_scatter

        node_att = 1.0 - node_att_aggregated
        return node_att


    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if att_log_logit.numel() == 0:
             return torch.tensor([], device=att_log_logit.device)

        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if emb.numel() == 0:
             if self.learn_edge_att:
                 return torch.empty((0, 1), device=emb.device)
             else:
                 return torch.empty((0, 1), device=emb.device)


        if self.learn_edge_att:
            if edge_index.numel() == 0:
                 return torch.empty((0, 1), device=emb.device)
            if edge_index.max() >= emb.size(0):
                 print(f"Warning: Invalid edge index in ExtractorMLP for emb size {emb.size(0)}")
                 return torch.empty((edge_index.size(1), 1), device=emb.device)


            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            batch_indices = batch[col] if batch is not None and batch.numel() > 0 else None
            att_log_logits = self.feature_extractor(f12, batch_indices)
        else:
            batch_indices = batch if batch is not None and batch.numel() > 0 else None
            att_log_logits = self.feature_extractor(emb, batch_indices)
        return att_log_logits

import sys
import matplotlib.pyplot as plt

def train_xgnn_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, args):
    print('=' * 40)
    print('=' * 40)
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')
    print(f'[INFO] Meta-learning enabled: {args.meta}')
    print(f'[INFO] GMT Variant (multi_linear): {local_config[f"{method_name}_config"].get("multi_linear", "Default")}')


    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    try:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
            data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False)
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    model_config['deg'] = aux_info['deg']
    model_config['x_dim'] = x_dim
    model_config['edge_attr_dim'] = edge_attr_dim
    model_config['multi_label'] = aux_info['multi_label']
    try:
        model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None

    print('=' * 40)
    print('=' * 40)

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        pre_model_base = f"{data_dir}/{dataset_name}/" + model_name
        if args.num_layers > 0:
            pre_model_base += f"{args.num_layers}L"
        pre_model_name = pre_model_base + f'_seed{random_state}.pt'
        try:
            print(f'[INFO] Attempting to load a pre-trained model from {pre_model_name}')
            checkpoint = torch.load(pre_model_name, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'[INFO] Loaded pre-trained model.')
            if args.force_train:
                print("[INFO] Forced re-pretraining enabled.")
                raise FileNotFoundError
        except (FileNotFoundError, KeyError, Exception) as e:
            print(f'[INFO] Pre-trained model not found or error loading ({e}). Pretraining...')
            try:
                train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                                   model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
                save_checkpoint(model, model_dir=f"{data_dir}/{dataset_name}/", model_name=f'{model_name}{args.num_layers if args.num_layers > 0 else ""}L_seed{random_state}')
                print(f'[INFO] Finished pretraining and saved to {pre_model_name}')
            except Exception as pretrain_e:
                print(f"Error during pretraining: {pretrain_e}")
                return None, None
    else:
        print('[INFO] Training from scratch (no pretraining).')

    mt = args.gcat_multi_linear if args.gcat_multi_linear >= 0 else method_config.get('multi_linear', 3)
    ie = method_config['info_loss_coef']
    r = method_config['final_r']
    dr = method_config['decay_r']
    di = method_config['decay_interval']
    st = 100
    model_save_dir = data_dir / dataset_name / f'{args.log_dir}'
    model_save_dir.mkdir(parents=True, exist_ok=True)
    pre_model_name_mcmc = f"{dataset_name}_mt{mt}_{model_name}_scracth{method_config['from_scratch']}_ie{ie}_r{r}dr{dr}di{di}st{st}"
    if args.epochs > 0:
        pre_model_name_mcmc += f"ep{args.epochs}"
    pre_model_name_mcmc += f"sd{random_state}"

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)

    if method_config['from_mcmc']:
        pred_model_name_clf = model_save_dir / f"{pre_model_name_mcmc}_clf_mcmc.pt"
        pred_model_name_att = model_save_dir / f"{pre_model_name_mcmc}_att_mcmc.pt"
        print(f'[INFO] Attempting to load MCMC models: {pred_model_name_clf}, {pred_model_name_att}')
        try:
            load_checkpoint(model, model_save_dir, model_name=f"{pre_model_name_mcmc}_clf_mcmc")
            load_checkpoint(extractor, model_save_dir, model_name=f"{pre_model_name_mcmc}_att_mcmc")
            print('[INFO] Loaded models from MCMC checkpoints.')
        except Exception as e:
            print(f'[ERROR] Failed to load MCMC models: {e}. Continuing without MCMC loading.')
            method_config['from_mcmc'] = False


    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if not scheduler_config else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config, **method_config, 'meta_learning': args.meta}
    hparam_dict_log = {}
    for k, v in hparam_dict.items():
         if isinstance(v, (dict, list)):
             try:
                 hparam_dict_log[k] = str(v)
             except Exception:
                 hparam_dict_log[k] = "Cannot convert to string"
         else:
             hparam_dict_log[k] = v

    metric_dict = deepcopy(init_metric_dict)

    print('=' * 40)
    print(f'[INFO] Initializing GSAT (variant {method_config.get("multi_linear", "Default")})...')
    method_config_new = copy.deepcopy(method_config)
    method_config_new['mcmc_dir'] = model_save_dir
    method_config_new['pre_model_name'] = pre_model_name_mcmc

    try:
        xgnn = GSAT(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config_new, shared_config, model_config)
    except Exception as e:
        print(f"Error initializing GSAT class: {e}")
        return hparam_dict, metric_dict

    print(f'[INFO] Starting training (Meta: {args.meta})...')
    final_metric_dict = {}
    try:
        if args.meta:
            if 'higher' not in sys.modules:
                 print("Error: 'higher' library not installed. Cannot run meta-learning.")
                 final_metric_dict = metric_dict
            else:
                 print("[INFO] Running Meta-Training Loop...")
                 final_metric_dict = xgnn.train_self_meta(
                     loaders, test_set, metric_dict, model_config.get('use_edge_attr', True),
                     inner_steps=method_config.get('meta_inner_steps', 3),
                     adapt_steps=method_config.get('meta_adapt_steps', 3),
                     alpha=method_config.get('meta_alpha', 0.01),
                     hops=method_config.get('meta_hops', 1)
                 )
        else:
            print("[INFO] Running Standard Training Loop...")
            final_metric_dict = xgnn.train_self(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    except Exception as train_e:
        print(f"Error during training: {train_e}")
        final_metric_dict = metric_dict

    writer.close()
    return hparam_dict, final_metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT with optional Meta-Learning')
    parser.add_argument('--dataset', type=str, required=True, help='dataset used')
    parser.add_argument('--backbone', type=str, required=True, help='backbone model used (e.g., GIN, GCN)')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device id, -1 for cpu')
    parser.add_argument('-ld', '--log_dir', default='logs', type=str, help='Base directory for logs and saved models')
    parser.add_argument('-mt', '--multi_linear', default=-1, type=int, help='Which GMT variant to use (e.g., 3 for LIN, 8 for SAM). Overrides config.')
    parser.add_argument('-gmt', '--gcat_multi_linear', default=-1, type=int, help='Variant identifier for naming purposes (if different from -mt)')
    parser.add_argument('-st', '--sampling_trials', default=-1, type=int, help='Number of sampling rounds for GMT-SAM. Overrides config.')
    parser.add_argument('-fs', '--from_scratch', default=-1, type=int, help='Force training from scratch (1) or allow pretraining (0). Overrides config.')
    parser.add_argument('-fm', '--from_mcmc', action='store_true', help='Load model from saved MCMC checkpoint.')
    parser.add_argument('-sm', '--save_mcmc', action='store_true', help='Save model checkpoint at best validation epoch.')
    parser.add_argument('-sd', '--seed', default=-1, type=int, help='Specific random seed to use. If -1, run multiple seeds from global config.')
    parser.add_argument('-ep', '--epochs', default=-1, type=int, help='Number of training epochs. Overrides config.')
    parser.add_argument('-ft', '--force_train', action='store_true', help='Force re-pretraining even if a checkpoint exists.')
    parser.add_argument('-ie', '--info_loss_coef', default=-1, type=float, help='Coefficient for information loss. Overrides config.')
    parser.add_argument('-r', '--ratio', default=-1, type=float, help='Final target sparsity ratio (final_r). Overrides config.')
    parser.add_argument('-ir', '--init_r', default=-1, type=float, help='Initial sparsity ratio (init_r). Overrides config.')
    parser.add_argument('-sr', '--sel_r', default=-1, type=float, help='Selection ratio for subgraph decoding variants. Overrides config.')
    parser.add_argument('-dr', '--decay_r', default=-1, type=float, help='Sparsity ratio decay amount per interval. Overrides config.')
    parser.add_argument('-di', '--decay_interval', default=-1, type=int, help='Epoch interval for sparsity ratio decay. Overrides config.')
    parser.add_argument('-L', '--num_layers', default=-1, type=int, help='Number of GNN layers. Overrides config.')
    parser.add_argument('--meta', action='store_true', help='If set, perform meta-learning updates.')

    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSAT'

    print('=' * 40)
    print('=' * 40)
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print(f'[INFO] Meta-Learning: {args.meta}')
    print('=' * 40)

    try:
        global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
        local_config_name = get_local_config_name(model_name, dataset_name)
        local_config = yaml.safe_load((config_dir / local_config_name).open('r'))
    except FileNotFoundError as e:
        print(f"Error loading config file: {e}. Make sure config files exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing config files: {e}")
        sys.exit(1)


    method_cfg = local_config[f'{method_name}_config']
    model_cfg = local_config['model_config']

    if args.epochs >= 0: method_cfg['epochs'] = args.epochs
    if args.multi_linear >= 0: method_cfg['multi_linear'] = args.multi_linear
    if args.sampling_trials >= 0: method_cfg['sampling_trials'] = args.sampling_trials
    if args.from_scratch >= 0: method_cfg['from_scratch'] = bool(args.from_scratch)
    method_cfg['from_mcmc'] = args.from_mcmc
    method_cfg['save_mcmc'] = args.save_mcmc

    if args.info_loss_coef >= 0: method_cfg['info_loss_coef'] = args.info_loss_coef
    if args.ratio >= 0: method_cfg['final_r'] = args.ratio
    if args.init_r >= 0: method_cfg['init_r'] = args.init_r
    if args.sel_r >= 0: method_cfg['sel_r'] = args.sel_r
    if args.decay_r >= 0: method_cfg['decay_r'] = args.decay_r
    if args.decay_interval >= 0: method_cfg['decay_interval'] = args.decay_interval
    if args.num_layers >= 0: model_cfg['num_layers'] = args.num_layers

    print("--- Method Config ---")
    print(yaml.dump(method_cfg))
    print("--- Model Config ---")
    print(yaml.dump(model_cfg))
    print("--------------------")

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']
    run_seeds = range(num_seeds) if args.seed < 0 else [args.seed]

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 and torch.cuda.is_available() else 'cpu')

    metric_dicts = []
    hparam_dict_final = {}

    for r_state in run_seeds:
        print(f"\n===== Running Seed: {r_state} =====")
        log_suffix = f"{time_stamp}-{dataset_name}-{model_name}-seed{r_state}-{method_name}"
        log_suffix += f"-mt{method_cfg.get('multi_linear', 'D')}"
        log_suffix += f"-{'meta' if args.meta else 'std'}"
        log_dir_run = data_dir / dataset_name / f'{args.log_dir}' / log_suffix
        print(f"Log Directory: {log_dir_run}")

        hparam_dict, metric_dict = train_xgnn_one_seed(
            local_config, data_dir, log_dir_run, model_name, dataset_name, method_name, device, r_state, args
        )

        if metric_dict is not None:
             metric_dicts.append(metric_dict)
             if not hparam_dict_final:
                 hparam_dict_final = hparam_dict
        else:
             print(f"!!!!! Seed {r_state} failed !!!!!")


    print("\n===== Final Results =====")
    if not metric_dicts:
        print("No successful runs completed.")
        return

    final_metrics_agg = {}
    metric_keys = metric_dicts[0].keys()

    for key in metric_keys:
        metric_values = [m[key] for m in metric_dicts if m is not None and key in m and m[key] is not None]

        if not metric_values:
             print(f"{key}: N/A (No valid data)")
             continue

        metric_values_np = np.array(metric_values)
        mean_val = np.mean(metric_values_np)
        std_val = np.std(metric_values_np)
        final_metrics_agg[f"{key}_mean"] = mean_val
        final_metrics_agg[f"{key}_std"] = std_val

        scale = 100 if 'loss' not in key.lower() and 'epoch' not in key.lower() and 'tvd' not in key.lower() else 1
        print(f"{key}: {mean_val * scale:.2f} +/- {std_val * scale:.2f}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    main()
