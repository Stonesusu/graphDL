"""
Script that reads from raw device data and dumps into a pickle
file the following:

* A heterogeneous graph with categorical features.
* A list with all the movie titles.  The movie titles correspond to
  the movie nodes in the heterogeneous graph.

This script exemplifies how to prepare tabular data with textual
features.  Since DGL graphs do not store variable-length features, we
instead put variable-length features into a more suitable container
(e.g. torchtext to handle list of texts)
"""

import os
import json
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from data_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=str)
    parser.add_argument('output_path',type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path

    #load data
    users = pd.read_csv(os.path.join(directory,'user.csv'))
    users = users[['customer_id','trans_time','target']]
    #按交易时间排序
    users = users.sort_values(['trans_time'])
    
    wifi_list = pd.read_csv(os.path.join(directory,'wifi_list_id.csv'))
    wifi_list = wifi_list[['customer_id','create_time','wifi_list']]
    wifi_list = wifi_list.sort_values(['customer_id','wifi_list','create_time'])
    wifi_list = wifi_list.drop_duplicates(['customer_id','wifi_list'])
    wifi_list = wifi_list.rename(columns={'wifi_list':'wifi_id'})
    
    user_wifi_list = pd.merge(users,wifi_list,how='inner',on='customer_id')
    
    # Filter the users and items that never appear in the rating table.
    distinct_users_in_relation = user_wifi_list['customer_id'].unique()
    distinct_wifi_in_relation = user_wifi_list['wifi_id'].unique()
    users = users[users['customer_id'].isin(distinct_users_in_relation)]
    wifi_list = wifi_list[wifi_list['wifi_id'].isin(distinct_wifi_in_relation)]
#     print(users)
#     print(wifi_list)
    
    # Build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'customer_id', 'user')
    graph_builder.add_entities(user_wifi_list.drop_duplicates('wifi_id'), 'wifi_id', 'wifi')
    graph_builder.add_binary_relations(user_wifi_list, 'customer_id', 'wifi_id', 'relation')
    graph_builder.add_binary_relations(user_wifi_list, 'wifi_id', 'customer_id', 'relation-by')

    g = graph_builder.build()
    
    print(g)
#     print(g.nodes('user'))
#     print(g.nodes('wifi'))
#     print(g.edges(etype='relation'))
#     print(g.edges(etype='relation-by'))

    g.nodes['user'].data['label'] = torch.tensor(users['target'].values)
    
    g.nodes['user'].data['h_embedding'] = torch.nn.Embedding(users.shape[0],32).weight
    
#     g.edges['relation'].data['weights'] = torch.ones(g.num_edges('relation'))
#     g.edges['relation-by'].data['weights'] = torch.ones(g.num_edges('relation-by'))
    
    # Train-validation-test split
    user_train_indices,user_val_indices,user_test_indices,train_indices,val_indices,test_indices = train_test_split_by_time1(users,user_wifi_list, 'trans_time', 'customer_id')
    
#     train_g = build_train_graph(g, train_indices, 'user', 'wifi', 'relation', 'relation-by')
#     print(train_g)
    
    dataset = {
#         'train-graph': train_g,
        'train-graph': g,
        'train-indices':user_train_indices,
        'val-indices':user_val_indices,
        'test-indices':user_test_indices,
        #'val-matrix': val_matrix,
        #'test-matrix': test_matrix,
        'item-texts': {},
        'item-images': None,
        'user-type': 'user',
        'item-type': 'wifi',
        'user-to-item-type': 'relation',
        'item-to-user-type': 'relation-by',
        'timestamp-edge-column': 'create_time'}

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    

print("Done!")