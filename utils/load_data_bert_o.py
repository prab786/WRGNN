import pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import numpy as np
import spacy
import pickle
from spacy.tokens import Doc
from transformers import BertTokenizer
import torch
from torch_geometric.data import Data
import os
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
DEP_RELATION_WEIGHTS = {
    "root": 1.0, "nsubj": 0.9, "dobj": 0.8, "amod": 0.7,
    "prep": 0.6, "pobj": 0.5, "det": 0.4, "advmod": 0.3, "other": 0.1
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

def construct_text_graph(text, tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Constructs a graph representation of a text using BERT tokens and spaCy dependencies.

    Args:
        text (str): Input text.
        tokenizer: BERT tokenizer.
        spacy_nlp: spaCy NLP model.

    Returns:
        torch_geometric.data.Data: Graph representation with edge index and edge weights.
        list: List of token IDs for BERT tokens.
    """
    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]
    spacy_dependencies = [(token.text, token.dep_, token.head.text, token.i, token.head.i) for token in doc]

    # Tokenize with BERT
    bert_tokens = tokenizer.tokenize(text)
    bert_token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

    # Map spaCy tokens to BERT subword indices
    mapping_spacy_to_bert = []
    current_bert_index = 0
    buffer = ''

    for spacy_idx, token in enumerate(spacy_tokens):
        subword_indices = []
        while current_bert_index < len(bert_tokens):
            bert_token = bert_tokens[current_bert_index]
            
            buffer += bert_token.replace("##", "")  # Remove '##' prefix for BERT subwords
            subword_indices.append(current_bert_index)
            current_bert_index += 1

            if buffer == token:
                buffer = ''  # Reset for next token
                break

        mapping_spacy_to_bert.append((spacy_idx, subword_indices))

    # Create BERT-based dependency graph
    dependency_graph_bert = []
    edges = []

    for (token, dep, head, token_idx, head_idx) in spacy_dependencies:
        token_bert_indices = [bert_idx for spacy_idx, bert_indices in mapping_spacy_to_bert if spacy_idx == token_idx for bert_idx in bert_indices]
        head_bert_indices = [bert_idx for spacy_idx, bert_indices in mapping_spacy_to_bert if spacy_idx == head_idx for bert_idx in bert_indices]

        for t_bert_idx in token_bert_indices:
            for h_bert_idx in head_bert_indices:
                dependency_graph_bert.append({
                    "token": token,
                    "dep": dep,
                    "head": head,
                    "token_bert_id": bert_token_ids[t_bert_idx],
                    "head_bert_id": bert_token_ids[h_bert_idx]
                })
                edges.append([t_bert_idx, h_bert_idx])

    # Construct adjacency matrix for the graph
    ed = []
    ed_weights = []
    for i, token_id1 in enumerate(bert_token_ids):
        for j, token_id2 in enumerate(bert_token_ids):
            if ([token_id1, token_id2] in edges or token_id1 == token_id2) and i < 512 and j < 512:
                ed.append([i, j])
                ed_weights.append(1.0)
            if j >= 512:
                break
        if i >= 512:
            break

    # Convert edges to torch tensors
    edge_index = torch.tensor(ed, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(ed_weights, dtype=torch.float)

    # Construct the graph data object for torch-geometric
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight)

    return graph_data, bert_token_ids


from datasets import load_dataset, DatasetDict

def load_articles(obj="imdb"):
   
    dataset = load_dataset(obj)
    
    # The IMDB dataset already has predefined train and test splits, creating validation manually
    train_valid = dataset["train"].train_test_split(test_size=0.1)
    dataset = DatasetDict({
        "train": train_valid["train"],
        "validation": train_valid["test"],
        "test": dataset["test"]
    })
    
    x_train, y_train = dataset["train"]["text"], dataset["train"]["label"]
    x_test, y_test = dataset["validation"]["text"], dataset["validation"]["label"]
    x_test_res = dataset["test"]["text"]
    y_test_res = dataset["test"]["label"]
    
    # Load or construct text graphs (modify based on your implementation)
    x_train_graph = load_graphs("berttrain_"+obj+".Rograph", x_train)
    x_test_graph = load_graphs("berttest_"+obj+".Rograph", x_test)
    x_test_res_graph = load_graphs("berttest_res_a_"+obj+".Rograph", x_test_res)
    
    return x_train, x_test, x_test_res, y_train, y_test, y_test_res, x_train_graph, x_test_graph, x_test_res_graph



def file_exists(file_path):
    return os.path.exists(file_path)

def load_graphs(file_path,texts,):
        if(file_exists(file_path)):
            return pickle.load(open(file_path,'rb'))
        idx2graph = {}
        fout = open(file_path, 'wb')
        for i,ele in enumerate(texts):
            #print(i)
            adj_matrix,_ = construct_text_graph(ele[:600])
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
    
    x_train_res1_graph=load_graphs('bertx_train_res1'+obj+'.Rograph',x_train_res1)
    x_train_res1_2_graph=load_graphs('bertx_train_res1_2'+obj+'.Rograph',x_train_res1_2)
    x_train_res2_graph=load_graphs('bertx_train_res2'+obj+'.Rograph',x_train_res2)
    x_train_res2_2_graph=load_graphs('bertx_train_res2_2'+obj+'.Rograph',x_train_res2_2)

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
    
# x_train, x_test, x_test_res, y_train, y_test, y_test_res, x_train_graph, x_test_graph, x_test_res_graph=load_articles(obj)
# print(y_test_res)
