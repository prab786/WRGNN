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
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)
    )

    valid_mask = valid_mask.view(-1)

    return edge_index[:, valid_mask]

def align_spacy_to_roberta(text, tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Aligns spaCy tokens with RoBERTa subwords using offset mapping.
    """
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    roberta_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"]) 
    offset_mapping = encoded["offset_mapping"]

    special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id}
    valid_indices = [i for i, tok_id in enumerate(encoded["input_ids"]) if tok_id not in special_tokens]

    mapping_spacy_to_roberta = [[] for _ in spacy_tokens]

    spacy_idx = 0
    spacy_cursor = 0 

    for roberta_idx, (start, end) in enumerate(offset_mapping):
        if start == end or roberta_idx not in valid_indices:
            continue

        while spacy_idx < len(spacy_tokens) and spacy_cursor < start:
            spacy_cursor += len(spacy_tokens[spacy_idx]) + 1
            spacy_idx += 1

        if spacy_idx < len(spacy_tokens):
            mapping_spacy_to_roberta[spacy_idx - 1].append(roberta_idx)

    return mapping_spacy_to_roberta

def construct_text_graph(text, tokenizer=tokenizer, spacy_nlp=nlp):
    doc = spacy_nlp(text)    
    spacy_dependencies = [(token.text, token.dep_, token.head.text, token.i, token.head.i) for token in doc]
    
    roberta_encoding = tokenizer(text, return_offsets_mapping=True)
    roberta_tokens = roberta_encoding.tokens()
    roberta_token_ids = roberta_encoding.input_ids

    mapping_spacy_to_roberta = align_spacy_to_roberta(text, tokenizer, spacy_nlp)

    edges = []
    edge_weights = []

    for (token, dep, head, token_idx, head_idx) in spacy_dependencies:
        token_roberta_indices = mapping_spacy_to_roberta[token_idx] if token_idx < len(mapping_spacy_to_roberta) else []
        head_roberta_indices = mapping_spacy_to_roberta[head_idx] if head_idx < len(mapping_spacy_to_roberta) else []

        for t_roberta_idx in token_roberta_indices:
            for h_roberta_idx in head_roberta_indices:
                edges.append([t_roberta_idx, h_roberta_idx])
                edge_weights.append(DEP_RELATION_WEIGHTS.get(dep, DEP_RELATION_WEIGHTS["other"]))
    
    ed = []
    ed_weights = []
    actual_num_nodes = len(roberta_token_ids) - 1

    for i, token_id1 in enumerate(roberta_token_ids):
        for j, token_id2 in enumerate(roberta_token_ids):
            if ([i, j] in edges or i == j) and i < actual_num_nodes and j < actual_num_nodes:
                ed.append([i, j])
                ed_weights.append(edge_weights[edges.index([i, j])] if [i, j] in edges else 1.0)
            if j >= actual_num_nodes:
                break
        if i >= actual_num_nodes: 
            break

    edge_index = torch.tensor(ed, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(ed_weights, dtype=torch.float)
   
    valid_edge_mask = (edge_index[0] < actual_num_nodes) & (edge_index[1] < actual_num_nodes)
    edge_index = edge_index[:, valid_edge_mask]  
    edge_weight = edge_weight[valid_edge_mask]

    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight)
    
    return graph_data, roberta_token_ids

# ... rest of the code remains the same ...