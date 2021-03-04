import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import dgl
import tqdm

import layers
import sampler as sampler_module
# import evaluation

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, hidden_dims,output_dims,n_layers):
        super().__init__()

#         self.proj = layers.LinearProjector(full_graph, ntype, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims,output_dims,n_layers)
#         self.scorer = layers.ItemToItemScorer(full_graph, ntype)

#     def forward(self, pos_graph, neg_graph, blocks):
    def forward(self,blocks):
        h_item = self.get_repr(blocks)
#         pos_score = self.scorer(pos_graph, h_item)
#         neg_score = self.scorer(neg_graph, h_item)
#         return (neg_score - pos_score + 1).clamp(min=0)
        return h_item

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)

def train(dataset, args):
    g = dataset['train-graph']
    user_train_indices = dataset['train-indices']
    user_val_indices = dataset['val-indices']
    user_test_indices = dataset['test-indices']
#     val_matrix = dataset['val-matrix'].tocsr()
#     test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    item_to_user_type = dataset['item-to-user-type']
    timestamp = dataset['timestamp-edge-column']

    device = torch.device(args.device)


#     # Assign user and movie IDs and use them as features (to learn an individual trainable
#     # embedding for each entity)
#     g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
#     g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))

#     # Prepare torchtext dataset and vocabulary
#     fields = {}
#     examples = []
#     for key, texts in item_texts.items():
#         fields[key] = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
#     for i in range(g.number_of_nodes(item_ntype)):
#         example = torchtext.data.Example.fromlist(
#             [item_texts[key][i] for key in item_texts.keys()],
#             [(key, fields[key]) for key in item_texts.keys()])
#         examples.append(example)
#     textset = torchtext.data.Dataset(examples, fields)
#     for key, field in fields.items():
#         field.build_vocab(getattr(textset, key))
#         #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')

#     # Sampler
#     batch_sampler = sampler_module.ItemToItemBatchSampler(
#         g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = sampler_module.PinSAGECollator(neighbor_sampler, g, user_ntype)
    
#     #这个错误其实是pytorch函数torch.utils.data.DataLoader在windows下的特有错误，
#     #该函数里面有个参数num_workers表示进程个数，在windows下改为0就可以了
#     dataloader = DataLoader(
#         batch_sampler,
#         collate_fn=collator.collate_train,
#         num_workers=args.num_workers)
    dataloader = DataLoader(
        torch.arange(g.number_of_nodes(user_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(user_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)
    

    print('*************************************')
#     print(g.number_of_nodes(user_ntype))
    blocks,labels = next(dataloader_it)
    print(block.edata)
#     print(blocks[0].dstdata[dgl.NID])
#     print(blocks[0].srcdata[dgl.NID])
#     print(blocks[-1].dstdata[dgl.NID])
#     print(blocks[-1].srcdata[dgl.NID])
#     print(block.ntypes)
#     print(block.ntypes[0])
#     print('DST/' + block.ntypes[0])
#     print(block)
#     print(block.number_of_nodes('DST/' + block.ntypes[0]))
#     print(block.number_of_nodes('SRC/' + block.ntypes[0]))
#     print(block.number_of_nodes('XXX/' + block.ntypes[0]))
    
#     def forward(self, blocks, h):
#         for layer, block in zip(self.convs, blocks):
#             h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
#             h = layer(block, (h, h_dst), block.edata['weights'])
#         return h

    # Model
    model = PinSAGEModel(g, user_ntype, args.hidden_dims,args.output_dims ,args.num_layers).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # For each batch of blocks-labels
    for epoch_id in range(1):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
#             pos_graph, neg_graph, blocks = next(dataloader_it)
            blocks,labels = next(dataloader_it)
#             print(labels)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            labels.to(device)
# #             pos_graph = pos_graph.to(device)
# #             neg_graph = neg_graph.to(device)

            
#             logits = model(g,node_features)
#             loss = model(pos_graph, neg_graph, blocks).mean()
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#         # Evaluate
#         model.eval()
#         with torch.no_grad():
#             item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args.batch_size)
#             h_item_batches = []
#             for blocks in dataloader_test:
#                 for i in range(len(blocks)):
#                     blocks[i] = blocks[i].to(device)

#                 h_item_batches.append(model.get_repr(blocks))
#             h_item = torch.cat(h_item_batches, 0)

#             print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--output-dims', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batches-per-epoch', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(dataset)
    train(dataset, args)
