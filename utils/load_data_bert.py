import pickle
import numpy as np
import pandas as pd
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
from transformers import AutoTokenizer
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
graph_counter = 0
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
#bert_model = BertModel.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased")
DEP_RELATION_WEIGHTS = {
    "ROOT": 1.0, "nsubj": 0.9, "dobj": 0.8, "amod": 0.7,
    "prep": 0.6, "pobj": 0.5, "det": 0.4, "advmod": 0.3, "other": 0.1
}
DEP_RELATION_TYPE = {
    "ROOT": 0, "nsubj":1, "dobj":2, "amod": 3,
    "prep": 4, "pobj": 5, "det": 6, "advmod": 7, "other": 8
}
def clean_edge_index3(edge_index,edge_type ,edge_batch=0, num_nodes=511):
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
    #print(3)
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    # Generate a boolean mask for each edge being within valid bounds for its batch
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)     ) #=

    # Reshape valid_mask to 1D to match the second dimension of edge_index
    valid_mask = valid_mask.view(-1)
    # print(valid_mask.shape)
    # print(edge_index.shape)
    # print(edge_type.shape)
    # Apply the mask to filter the edges
    return edge_index[:, valid_mask],edge_type[valid_mask]

def clean_edge_index2(edge_index,edge_type ,edge_batch=0, num_nodes=511):
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
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)     ) #=

    # Reshape valid_mask to 1D to match the second dimension of edge_index
    valid_mask = valid_mask.view(-1)
    # print(valid_mask.shape)
    # print(edge_index.shape)
    # print(edge_type.shape)
    # Apply the mask to filter the edges
    return edge_index[:, valid_mask],edge_type[valid_mask]

def clean_edge_index1(edge_index,edge_type ,edge_batch=0, num_nodes=511):
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
    #print(1)
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    # Generate a boolean mask for each edge being within valid bounds for its batch
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)  ) #= & (edge_type != 8 )

    # Reshape valid_mask to 1D to match the second dimension of edge_index
    valid_mask = valid_mask.view(-1)
    # print(valid_mask.shape)
    # print(edge_index.shape)
    # print(edge_type.shape)
    # Apply the mask to filter the edges
    return edge_index[:, valid_mask],edge_type[valid_mask]
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

def map_spacy_to_bert(text, tokenizer, spacy_nlp):
    """
    Maps spaCy tokens to BERT subword indices using character-based alignment.

    Args:
        text (str): Input text.
        tokenizer: BERT tokenizer.
        spacy_nlp: spaCy NLP model.

    Returns:
        list: List of tuples (spacy_token_index, [bert_subword_indices]).
    """

    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]

    # Tokenize with BERT and get character offsets
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    bert_tokens = encoded.tokens()
    offsets = encoded.offset_mapping  # Each subword's (start, end) character positions

    # Map each character position to its corresponding BERT token index
    char_to_bert = {}
    for bert_idx, (start, end) in enumerate(offsets):
        for char_idx in range(start, end):  
            char_to_bert[char_idx] = bert_idx

    # Map spaCy tokens to corresponding BERT subwords
    mapping_spacy_to_bert = []
    for spacy_idx, token in enumerate(doc):
        token_start = token.idx  # Start position in text
        token_end = token_start + len(token)

        # Collect all BERT subword indices that align with this token
        subword_indices = sorted(set(char_to_bert.get(i) for i in range(token_start, token_end) if i in char_to_bert))

        mapping_spacy_to_bert.append((spacy_idx, subword_indices))

    return mapping_spacy_to_bert


def construct_text_graph(text, save_path=None,tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Constructs a graph representation of a text using BERT tokens and spaCy dependencies.

    Args:
        text (str): Input text.
        tokenizer: BERT tokenizer.
        spacy_nlp: spaCy NLP model.
        save_path (str): File path to save the graph information.

    Returns:
        torch_geometric.data.Data: Graph representation with edge index.
        list: List of token IDs for BERT tokens.
        pd.DataFrame: DataFrame containing graph edges and their dependency relations.
    """
    global graph_counter

    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_dependencies = [(token.i, token.dep_, token.head.i) for token in doc]  # (token_idx, dep, head_idx)

    # Tokenize with BERT
    bert_tokens = tokenizer.tokenize(text)
    bert_token_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

    # Get mapping from spaCy tokens to BERT subwords
    mapping_spacy_to_bert = map_spacy_to_bert(text, tokenizer, spacy_nlp)

    # Create BERT-based dependency graph
    edges = []
    edge_weights=[]
    dependency_relations = []
    #print(spacy_dependencies)
    for token_idx, dep, head_idx in spacy_dependencies:
        token_bert_indices = [bert_idx for _, bert_indices in mapping_spacy_to_bert if _ == token_idx for bert_idx in bert_indices]
        head_bert_indices = [bert_idx for _, bert_indices in mapping_spacy_to_bert if _ == head_idx for bert_idx in bert_indices]

        for t_bert_idx in token_bert_indices:
            for h_bert_idx in head_bert_indices:
                edges.append([t_bert_idx, h_bert_idx])
                dependency_relations.append(DEP_RELATION_TYPE.get(dep, DEP_RELATION_TYPE["other"]))                              
                edge_weights.append(DEP_RELATION_WEIGHTS.get(dep, DEP_RELATION_WEIGHTS["other"]))  # You can modify this if needed
                
    # Convert edges to torch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    edge_type = torch.tensor(dependency_relations, dtype=torch.long)

    # Construct the graph data object for torch-geometric
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight, edge_type=edge_type)
    df=[]
    # Construct the graph data object for torch-geometric
   # graph_data = Data(edge_index=edge_index)

    # Store edge data in DataFrame
    # df = pd.DataFrame({
    #     "graph_id": graph_counter,
    #     "source_node": graph_data.edge_index[0].tolist(),
    #     "target_node": graph_data.edge_index[1].tolist(),
    #     "dependency_relation": dependency_relations
    # })

    # # Save DataFrame to CSV (append if file exists)
    # if save_path != None:
    #     df.to_csv(save_path, mode='a', index=False, header=not pd.io.common.file_exists(save_path))

    graph_counter += 1  # Increment graph ID

    return graph_data, bert_token_ids, df



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
    x_trian_graph=load_graphs('gr/nnberttrain'+obj+'.Rograph',x_train)
    x_test_graph=load_graphs('gr/nnberttest'+obj+'.Rograph',x_test)
    x_test_res_graph=load_graphs('gr/nnberttest_res__a1'+obj+'.Rograph',x_test_res)
    return x_train, x_test, x_test_res, y_train, y_test,x_trian_graph,x_test_graph,x_test_res_graph


def file_exists(file_path):
    return os.path.exists(file_path)

def load_graphs(file_path,texts):
        if(file_exists(file_path)):
            return pickle.load(open(file_path,'rb'))
        idx2graph = {}
        fout = open(file_path, 'wb')
        for i,ele in enumerate(texts):
            
            #print(i)
            adj_matrix,_,_ = construct_text_graph(ele[:2000])
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
    
    x_train_res1_graph=load_graphs('gr/nnbertx_train_res1'+obj+'.Rograph',x_train_res1)
    x_train_res1_2_graph=load_graphs('gr/nnbertx_train_res1_2'+obj+'.Rograph',x_train_res1_2)
    x_train_res2_graph=load_graphs('gr/nnbertx_train_res2'+obj+'.Rograph',x_train_res2)
    x_train_res2_2_graph=load_graphs('gr/nnbertx_train_res2_2'+obj+'.Rograph',x_train_res2_2)

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
    
#load_articles("politifact")
#a,b,c=construct_text_graph("You can modify this if needed")