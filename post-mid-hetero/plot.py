import torch
from torchviz import make_dot
from simple_motif import ExplainableHeteroGNN  # adjust path

# Dummy input matching your model
x_dict = {
    'author': torch.randn(8, 64, requires_grad=True),
    'paper': torch.randn(10, 64, requires_grad=True)
}
edge_index_dict = {
    ('author', 'writes', 'paper'): torch.randint(0, 8, (2, 20)),
    ('paper', 'cites', 'paper'): torch.randint(0, 10, (2, 15)),
    ('author', 'self_loop', 'author'): torch.stack([torch.arange(8), torch.arange(8)])
}
batch_dict = {
    'author': torch.randint(0, 2, (8,)),
    'paper': torch.randint(0, 2, (10,))
}

model = ExplainableHeteroGNN()
logits = model(x_dict, edge_index_dict, batch_dict)
make_dot(logits, params=dict(list(model.named_parameters()))).render("model_graph", format="png")

