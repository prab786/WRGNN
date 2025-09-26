import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import spacy
import pickle
from spacy.tokens import Doc
from transformers import RobertaTokenizer
import torch
from torch_geometric.data import Data
import os
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
#bert_model = BertModel.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
DEP_RELATION_WEIGHTS = {
    "root": 1.0, "nsubj": 0.9, "dobj": 0.8, "amod": 0.7,
    "prep": 0.1, "pobj": 0.5, "det": 0.41, "advmod": 0.41, "other": 0.5
}

def clean_edge_index(edge_index, edge_batch=0, num_nodes=511):
    """
    Cleans the edge_index tensor by removing edges that reference nodes
    outside the allowed range for each batch.

    Parameters:
    - edge_index (torch.Tensor): 2D tensor of shape [2, num_edges] representing the edges.
    - edge_batch (torch.Tensor): 1D tensor of shape [num_edges] indicating the batch for each edge.
    - num_nodes (int): Number of nodes per batch.

    Returns:
    - torch.Tensor: The filtered edge_index with edges only within batch boundaries.
    """
    # assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index should have shape [2, num_edges]."
    # assert edge_batch.dim() == 1 and edge_batch.size(0) == edge_index.size(1), "edge_batch should have shape [num_edges]."

    # Calculate the valid node range for each edge based on its batch
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    # Generate a boolean mask for each edge being within valid bounds for its batch
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)
    )

    # Reshape valid_mask to 1D to match the second dimension of edge_index
    valid_mask = valid_mask.view(-1)
    # print(valid_mask.shape)
    # print(edge_index.shape)
    # Apply the mask to filter the edges
    return edge_index[:, valid_mask]




def align_spacy_to_roberta(text, tokenizer=tokenizer, spacy_nlp=nlp):
   

    """
    Aligns spaCy tokens with RoBERTa subwords using offset mapping.
    """
    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]

    # Tokenize with RoBERTa and get offset mappings
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    roberta_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offset_mapping = encoded["offset_mapping"]

    # Remove special tokens <s> and </s>
    special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id}
    valid_indices = [i for i, tok_id in enumerate(encoded["input_ids"]) if tok_id not in special_tokens]

    # Initialize mapping
    mapping_spacy_to_roberta = [[] for _ in spacy_tokens]

    spacy_idx = 0
    spacy_cursor = 0  # Tracks character positions in spaCy tokens

    for roberta_idx, (start, end) in enumerate(offset_mapping):
        if start == end or roberta_idx not in valid_indices:
            continue  # Skip special tokens

        while spacy_idx < len(spacy_tokens) and spacy_cursor < start:
            spacy_cursor += len(spacy_tokens[spacy_idx]) + 1  # +1 for spaces
            spacy_idx += 1

        if spacy_idx < len(spacy_tokens):
            mapping_spacy_to_roberta[spacy_idx - 1].append(roberta_idx)  # Assign subword index

    return mapping_spacy_to_roberta



def construct_text_graph(text, tokenizer=tokenizer, spacy_nlp=nlp):
    # Process text with spaCy
    doc = spacy_nlp(text)    
    spacy_dependencies = [(token.text, token.dep_, token.head.text, token.i, token.head.i) for token in doc]
    
    # Tokenize with RoBERTa
    roberta_encoding = tokenizer(text, return_offsets_mapping=True)
    roberta_tokens = roberta_encoding.tokens()
    roberta_token_ids = roberta_encoding.input_ids
    #print(roberta_tokens)
    # Map spaCy tokens to RoBERTa subword indices
    mapping_spacy_to_roberta = align_spacy_to_roberta(text, tokenizer, spacy_nlp)
    #print(mapping_spacy_to_roberta)
    # Create RoBERTa-based dependency graph
    edges = []

    #print(spacy_dependencies)
    for (token, dep, head, token_idx, head_idx) in spacy_dependencies:
        token_roberta_indices = mapping_spacy_to_roberta[token_idx] if token_idx < len(mapping_spacy_to_roberta) else []
        head_roberta_indices = mapping_spacy_to_roberta[head_idx] if head_idx < len(mapping_spacy_to_roberta) else []

        for t_roberta_idx in token_roberta_indices:
            for h_roberta_idx in head_roberta_indices:
                edges.append([t_roberta_idx, h_roberta_idx])
    
    # Construct adjacency matrix for the graph
    ed = []
    ed_weights = []
    for i, token_id1 in enumerate(roberta_token_ids):
        for j, token_id2 in enumerate(roberta_token_ids):
            if ([i, j] in edges or i == j) and i < 512 and j < 512:
                ed.append([i, j])
                ed_weights.append(1.0)
            if j >= 512:
                break
        if i >= 512:
            break

    # Convert edges to torch tensors
    edge_index = torch.tensor(ed, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(ed_weights, dtype=torch.float)
    #print(edge_index)
    # Construct the graph data object for torch-geometric
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight)
    
    return graph_data, roberta_token_ids




# def construct_text_graph(text, tokenizer=tokenizer, spacy_nlp=nlp):
#     # Process text with spaCy
#     doc = spacy_nlp(text)    
#     spacy_tokens = [token.text for token in doc]
#     spacy_dependencies = [(token.text, token.dep_, token.head.text, token.i, token.head.i) for token in doc]
    
#     # Tokenize with RoBERTa
#     encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
#     roberta_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
#     offset_mapping = encoded["offset_mapping"]  # Maps token spans to original text

#     # Ignore special tokens (<s>, </s>)
#     special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id}
#     valid_indices = [i for i, tok_id in enumerate(encoded["input_ids"]) if tok_id not in special_tokens]

#     # Mapping spaCy tokens to RoBERTa subword indices
#     mapping_spacy_to_roberta = [[] for _ in spacy_tokens]
#     current_spacy_idx = 0
#     spacy_cursor = 0  # Tracks character positions in spaCy tokens

#     for roberta_idx, (start, end) in enumerate(offset_mapping):
#         if start == end or roberta_idx not in valid_indices:
#             continue  # Skip special tokens

#         while current_spacy_idx < len(spacy_tokens) and spacy_cursor < start:
#             spacy_cursor += len(spacy_tokens[current_spacy_idx])
#             current_spacy_idx += 1

#         if current_spacy_idx < len(spacy_tokens):
#             mapping_spacy_to_roberta[current_spacy_idx].append(roberta_idx)

#     # Debugging output
#     print(f"spaCy Tokens: {spacy_tokens}")
#     print(f"RoBERTa Tokens: {roberta_tokens}")
#     print(f"Mapping: {mapping_spacy_to_roberta}")

#     # Create RoBERTa-based dependency graph
#     edges = []
    
#     for token, dep, head, token_idx, head_idx in spacy_dependencies:
#         token_roberta_indices = mapping_spacy_to_roberta[token_idx] if token_idx < len(mapping_spacy_to_roberta) else []
#         head_roberta_indices = mapping_spacy_to_roberta[head_idx] if head_idx < len(mapping_spacy_to_roberta) else []

#         for t_idx in token_roberta_indices:
#             for h_idx in head_roberta_indices:
#                 edges.append([t_idx, h_idx])

#     # Construct adjacency matrix
#     ed = []
#     ed_weights = []
#     num_tokens = len(roberta_tokens)

#     for i in range(num_tokens):
#         for j in range(num_tokens):
#             if ([i, j] in edges or i == j) and i < 512 and j < 512:
#                 ed.append([i, j])
#                 ed_weights.append(1.0)
#             if j >= 512:
#                 break
#         if i >= 512:
#             break

#     # Convert to torch tensors
#     edge_index = torch.tensor(ed, dtype=torch.long).t().contiguous()
#     edge_weight = torch.tensor(ed_weights, dtype=torch.float)

#     # Construct graph data object
#     graph_data = Data(edge_index=edge_index, edge_weight=edge_weight)
    
#     return graph_data, encoded["input_ids"]


def load_articles(obj):
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))

    restyle_dict = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_A.pkl', 'rb'))
    # alternatively, switch to loading other adversarial test sets with '_test_adv_[B/C/D].pkl'

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']

    x_test_res = restyle_dict['news']
    # idx2graph = {}
    # fout = open('test_res'+obj+'.Rograph', 'wb')
    # for i,ele in enumerate(x_test_res):
    #     #print(i)
    #     adj_matrix,_ = construct_text_graph(ele[:600])
    #     idx2graph[i] = adj_matrix
    # pickle.dump(idx2graph, fout)        
    # fout.close()
    x_trian_graph=load_graphs('graph/train_t_all_'+obj+'.Rograph',x_train)
    x_test_graph=load_graphs('graph/test_t_all_'+obj+'.Rograph',x_test)
    x_test_res_graph=load_graphs('graph/test_t_all_res_a_'+obj+'.Rograph',x_test_res)
    return x_train, x_test, x_test_res, y_train, y_test,x_trian_graph,x_test_graph,x_test_res_graph


def file_exists(file_path):
    return os.path.exists(file_path)

def load_graphs(file_path,texts,):
        if(file_exists(file_path)):
            return pickle.load(open(file_path,'rb'))
        idx2graph = {}
        fout = open(file_path, 'wb')
        for i,ele in enumerate(texts):
            #print(i)
            adj_matrix,_ = construct_text_graph(ele[:2500])
            idx2graph[i] = adj_matrix
        pickle.dump(idx2graph, fout)        
        fout.close()
        return pickle.load(open(file_path,'rb'))

def load_reframing(obj):
    print("loading news augmentations")
    print('Dataset: ', obj)

    restyle_dict_train1_1 = pickle.load(open('data/reframings/' + obj+ '_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open('data/reframings/' + obj+ '_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open('data/reframings/' + obj+ '_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open('data/reframings/' + obj+ '_train_sensational.pkl', 'rb'))
    

    finegrain_dict1 = pickle.load(open('data/veracity_attributions/' + obj+ '_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(open('data/veracity_attributions/' + obj+ '_fake_standards_neutral_sensational.pkl', 'rb'))

    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])
    
    x_train_res1_graph=load_graphs('graph/x_train_res1_all_'+obj+'.Rograph',x_train_res1)
    x_train_res1_2_graph=load_graphs('graph/x_train_res1_all_2_'+obj+'.Rograph',x_train_res1_2)
    x_train_res2_graph=load_graphs('graph/x_train_res2_all_'+obj+'.Rograph',x_train_res2)
    x_train_res2_2_graph=load_graphs('graph/x_train_res2_all_2_'+obj+'.Rograph',x_train_res2_2)

    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]   
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]  
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]
    replace_idx = replace_idx.tolist()  # Convert to list if needed

    # Access and replace correctly
    for idx in replace_idx:
        x_train_res1_graph[idx] = x_train_res1_2_graph[idx]
        x_train_res2_graph[idx] = x_train_res2_2_graph[idx]

    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t,x_train_res1_graph,x_train_res2_graph
    
# a,b=construct_text_graph("We use the well-known fake news data repository FakeNewsNet which contains news articles along with their auxiliary information such as users metadata and news comments. ")
# print(a)
# print(b)