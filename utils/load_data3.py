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
    actual_num_nodes = len(roberta_token_ids) - 1  # Exclude CLS token
    for i, token_id1 in enumerate(roberta_token_ids):
        for j, token_id2 in enumerate(roberta_token_ids):
            # FIX: Use actual_num_nodes instead of 512
            if ([i, j] in edges or i == j) and i < actual_num_nodes and j < actual_num_nodes:
                ed.append([i, j])
                ed_weights.append(1.0)
            if j >= actual_num_nodes:  # FIX: Use actual limit
                break
        if i >= actual_num_nodes:  # FIX: Use actual limit
            break

    # Convert edges to torch tensors
    edge_index = torch.tensor(ed, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(ed_weights, dtype=torch.float)
   
    actual_num_nodes = len(roberta_token_ids) - 1  # Subtract CLS token
    
    # Filter out edges that reference non-existent nodes
    valid_edge_mask = (edge_index[0] < actual_num_nodes) & (edge_index[1] < actual_num_nodes)
    edge_index = edge_index[:, valid_edge_mask]
    edge_weight = edge_weight[valid_edge_mask]
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

def load_articles_all(obj):
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))

    restyle_dict = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_A.pkl', 'rb'))
    restyle_dictB = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_B.pkl', 'rb'))
    restyle_dictC = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_C.pkl', 'rb'))
    restyle_dictD = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_D.pkl', 'rb'))
    # alternatively, switch to loading other adversarial test sets with '_test_adv_[B/C/D].pkl'

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']

    x_test_res = restyle_dict['news']
    x_test_resB = restyle_dictB['news']
    x_test_resC = restyle_dictC['news']
    x_test_resD = restyle_dictD['news']
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
    x_test_res_graphb=load_graphs('graph/test_t_all_res_b_'+obj+'.Rograph',x_test_res)
    x_test_res_graphc=load_graphs('graph/test_t_all_res_c_'+obj+'.Rograph',x_test_res)
    x_test_res_graphd=load_graphs('graph/test_t_all_res_d_'+obj+'.Rograph',x_test_res)
    return  x_test_resB, x_test_resC, x_test_resD,x_test_res_graphb,x_test_res_graphc,x_test_res_graphd
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
    
def edge_to_node_graph(input_graph):
    """
    Transforms a graph where edges become nodes in the output graph.
    Two nodes in the output graph are connected if the corresponding edges
    in the input graph were neighbors (shared a vertex).
    
    Parameters:
    - input_graph (Data): PyTorch Geometric Data object with edge_index and optionally edge_weight
    
    Returns:
    - Data: New graph where input edges are nodes and neighboring edges are connected
    """
    edge_index = input_graph.edge_index
    edge_weight = input_graph.edge_weight if hasattr(input_graph, 'edge_weight') else None
    
    # Get number of edges in input graph
    num_edges = edge_index.shape[1]
    
    # Create a mapping from edge to its index
    edge_to_idx = {}
    edges_list = []
    
    for i in range(num_edges):
        edge = (edge_index[0, i].item(), edge_index[1, i].item())
        edge_to_idx[edge] = i
        edges_list.append(edge)
    
    # Find neighboring edges (edges that share a vertex)
    new_edges = []
    new_edge_weights = []
    
    for i in range(num_edges):
        edge1 = edges_list[i]
        src1, dst1 = edge1
        
        for j in range(i + 1, num_edges):
            edge2 = edges_list[j]
            src2, dst2 = edge2
            
            # Check if edges share a vertex
            if src1 == src2 or src1 == dst2 or dst1 == src2 or dst1 == dst2:
                # Add bidirectional edge in the new graph
                new_edges.append([i, j])
                new_edges.append([j, i])
                
                # If original graph had edge weights, combine them for the new edge
                if edge_weight is not None:
                    weight = (edge_weight[i] + edge_weight[j]) / 2.0
                    new_edge_weights.extend([weight, weight])
                else:
                    new_edge_weights.extend([1.0, 1.0])
    
    # Convert to tensor
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weights, dtype=torch.float)
    else:
        # Handle case with no edges
        new_edge_index = torch.tensor([[], []], dtype=torch.long)
        new_edge_weight = torch.tensor([], dtype=torch.float)
    
    # Create new graph data
    new_graph = Data(
        edge_index=new_edge_index,
        edge_weight=new_edge_weight,
        num_nodes=num_edges  # Number of nodes = number of edges in original graph
    )
    
    # Store original edge information as node features if needed
    # This creates a feature matrix where each node (originally an edge) 
    # stores its source and destination vertices
    node_features = torch.zeros((num_edges, 2), dtype=torch.long)
    for i, (src, dst) in enumerate(edges_list):
        node_features[i, 0] = src
        node_features[i, 1] = dst
    
    new_graph.x = node_features
    
    return new_graph


def edge_to_node_graph_directed(input_graph, consider_direction=True):
    """
    Advanced version that considers edge direction when determining neighbors.
    
    Parameters:
    - input_graph (Data): PyTorch Geometric Data object
    - consider_direction (bool): If True, only connect edges that follow directional flow
    
    Returns:
    - Data: Transformed graph
    """
    edge_index = input_graph.edge_index
    edge_weight = input_graph.edge_weight if hasattr(input_graph, 'edge_weight') else None
    
    num_edges = edge_index.shape[1]
    edges_list = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(num_edges)]
    
    new_edges = []
    new_edge_weights = []
    
    for i in range(num_edges):
        src1, dst1 = edges_list[i]
        
        for j in range(num_edges):
            if i == j:
                continue
                
            src2, dst2 = edges_list[j]
            
            if consider_direction:
                # Connect if edge i ends where edge j starts
                if dst1 == src2:
                    new_edges.append([i, j])
                    
                    if edge_weight is not None:
                        # Weight based on flow continuity
                        weight = edge_weight[i] * edge_weight[j]
                        new_edge_weights.append(weight)
                    else:
                        new_edge_weights.append(1.0)
            else:
                # Connect if edges share any vertex
                if src1 == src2 or src1 == dst2 or dst1 == src2 or dst1 == dst2:
                    new_edges.append([i, j])
                    
                    if edge_weight is not None:
                        weight = (edge_weight[i] + edge_weight[j]) / 2.0
                        new_edge_weights.append(weight)
                    else:
                        new_edge_weights.append(1.0)
    
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weights, dtype=torch.float)
    else:
        new_edge_index = torch.tensor([[], []], dtype=torch.long)
        new_edge_weight = torch.tensor([], dtype=torch.float)
    
    # Create node features storing original edge information
    node_features = torch.zeros((num_edges, 2), dtype=torch.long)
    for i, (src, dst) in enumerate(edges_list):
        node_features[i, 0] = src
        node_features[i, 1] = dst
    
    new_graph = Data(
        edge_index=new_edge_index,
        edge_weight=new_edge_weight,
        num_nodes=num_edges,
        x=node_features
    )
    
    return new_graph


def batch_edge_to_node_transform(graphs_dict):
    """
    Apply edge-to-node transformation to a dictionary of graphs.
    
    Parameters:
    - graphs_dict (dict): Dictionary mapping indices to graph Data objects
    
    Returns:
    - dict: Dictionary with transformed graphs
    """
    transformed_graphs = {}
    
    for idx, graph in graphs_dict.items():
        try:
            transformed_graph = edge_to_node_graph(graph)
            transformed_graphs[idx] = transformed_graph
        except Exception as e:
            print(f"Error transforming graph at index {idx}: {e}")
            # Keep original graph if transformation fails
            transformed_graphs[idx] = graph
    
    return transformed_graphs


# Example usage with the existing code structure
def transform_loaded_graphs(file_path, texts):
    """
    Load graphs and apply edge-to-node transformation.
    """
    # Load or create original graphs
    original_graphs = load_graphs(file_path, texts)
    
    # Transform all graphs
    transformed_file_path = file_path.replace('.Rograph', '_edge2node.Rograph')
    
    if file_exists(transformed_file_path):
        return pickle.load(open(transformed_file_path, 'rb'))
    
    transformed_graphs = batch_edge_to_node_transform(original_graphs)
    
    # Save transformed graphs
    with open(transformed_file_path, 'wb') as fout:
        pickle.dump(transformed_graphs, fout)
    
    return transformed_graphs