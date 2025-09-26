import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
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

# Define maximum sequence length for RoBERTa
MAX_SEQUENCE_LENGTH = 512

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
    """
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
    
    # Apply the mask to filter the edges
    return edge_index[:, valid_mask]

def truncate_text_for_roberta(text, tokenizer=tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """
    Truncate text to fit within RoBERTa's maximum sequence length.
    Accounts for special tokens (<s>, </s>).
    """
    # First tokenize to check length
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=False)
    
    if len(encoded['input_ids']) <= max_length:
        return text
    
    # If too long, truncate the text and re-tokenize
    # Account for special tokens by reserving space for them
    max_text_tokens = max_length - 2  # Reserve space for <s> and </s> tokens
    
    # Truncate and re-encode
    encoded_truncated = tokenizer(text, 
                                max_length=max_length, 
                                truncation=True, 
                                add_special_tokens=True,
                                return_offsets_mapping=True)
    
    # Get the character span of the truncated tokens (excluding special tokens)
    valid_tokens = [(i, offset) for i, offset in enumerate(encoded_truncated['offset_mapping']) 
                   if offset[0] != offset[1]]  # Exclude special tokens with (0,0) mapping
    
    if valid_tokens:
        # Get the end position of the last valid token
        last_token_end = valid_tokens[-1][1][1]
        truncated_text = text[:last_token_end]
        return truncated_text
    
    return text

def align_spacy_to_roberta(text, tokenizer=tokenizer, spacy_nlp=nlp, debug=False):
    """
    Aligns spaCy tokens with RoBERTa subwords using offset mapping.
    This function maps each spaCy token to the corresponding RoBERTa subword token(s).
    
    Args:
        text: Input text to align
        tokenizer: RoBERTa tokenizer
        spacy_nlp: spaCy language model
        debug: Whether to print debug information
    
    Returns:
        mapping_spacy_to_roberta: List where index i contains RoBERTa token indices for spaCy token i
        text: The (potentially truncated) text that was processed
    """
    # Truncate text first to ensure it fits in RoBERTa
    original_text = text
    text = truncate_text_for_roberta(text, tokenizer)
    
    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]

    # Tokenize with RoBERTa and get offset mappings
    encoded = tokenizer(text, 
                       return_offsets_mapping=True, 
                       add_special_tokens=True,
                       max_length=MAX_SEQUENCE_LENGTH,
                       truncation=True)
    
    roberta_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offset_mapping = encoded["offset_mapping"]

    # Identify special token positions
    special_token_positions = set()
    for i, tok_id in enumerate(encoded["input_ids"]):
        if tok_id in {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}:
            special_token_positions.add(i)

    # Initialize mapping: each spaCy token maps to a list of RoBERTa token indices
    mapping_spacy_to_roberta = [[] for _ in spacy_tokens]

    # Get character spans for each spaCy token using spaCy's built-in character indices
    spacy_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
    
    if debug:
        print(f"Original text length: {len(original_text)}, Processed text length: {len(text)}")
        print(f"Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        print(f"spaCy tokens: {[(i, token.text, spacy_spans[i]) for i, token in enumerate(doc)]}")
        print(f"RoBERTa tokens: {[(i, tok, offset_mapping[i]) for i, tok in enumerate(roberta_tokens) if i not in special_token_positions]}")
    
    # Align tokens based on character overlap
    for roberta_idx, (roberta_start, roberta_end) in enumerate(offset_mapping):
        # Skip special tokens and empty spans
        if roberta_idx in special_token_positions or roberta_start == roberta_end:
            continue
            
        # Find which spaCy tokens overlap with this RoBERTa token
        best_overlap = 0
        best_spacy_idx = -1
        
        for spacy_idx, (spacy_start, spacy_end) in enumerate(spacy_spans):
            # Check for character span overlap
            overlap_start = max(roberta_start, spacy_start)
            overlap_end = min(roberta_end, spacy_end)
            overlap_length = max(0, overlap_end - overlap_start)
            
            # Find the spaCy token with the most overlap
            if overlap_length > best_overlap:
                best_overlap = overlap_length
                best_spacy_idx = spacy_idx
        
        # Assign RoBERTa token to the best matching spaCy token
        if best_spacy_idx >= 0:
            mapping_spacy_to_roberta[best_spacy_idx].append(roberta_idx)
    
    # Handle unmapped spaCy tokens by finding closest RoBERTa tokens
    for spacy_idx, roberta_indices in enumerate(mapping_spacy_to_roberta):
        if not roberta_indices and spacy_idx < len(spacy_tokens):
            if debug:
                print(f"Warning: spaCy token '{spacy_tokens[spacy_idx]}' at {spacy_spans[spacy_idx]} has no RoBERTa mapping")
            
            # Try to find the closest RoBERTa token
            spacy_start, spacy_end = spacy_spans[spacy_idx]
            spacy_mid = (spacy_start + spacy_end) / 2
            
            closest_roberta_idx = -1
            min_distance = float('inf')
            
            for roberta_idx, (roberta_start, roberta_end) in enumerate(offset_mapping):
                if roberta_idx in special_token_positions or roberta_start == roberta_end:
                    continue
                    
                roberta_mid = (roberta_start + roberta_end) / 2
                distance = abs(roberta_mid - spacy_mid)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_roberta_idx = roberta_idx
            
            if closest_roberta_idx >= 0:
                mapping_spacy_to_roberta[spacy_idx].append(closest_roberta_idx)
                if debug:
                    print(f"  -> Mapped to closest RoBERTa token {closest_roberta_idx}")
    
    if debug:
        print("Final mapping:")
        for i in range(len(spacy_tokens)):
            print(f"  spaCy[{i}] '{spacy_tokens[i]}' -> RoBERTa{mapping_spacy_to_roberta[i]}")
    
    return mapping_spacy_to_roberta, text

def construct_text_graph(text, tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Construct text graph with proper truncation handling for RoBERTa.
    """
    # Truncate text first
    text = truncate_text_for_roberta(text, tokenizer)
    
    # Process text with spaCy
    doc = spacy_nlp(text)    
    spacy_dependencies = [(token.text, token.dep_, token.head.text, token.i, token.head.i) for token in doc]
    
    # Tokenize with RoBERTa with proper truncation
    roberta_encoding = tokenizer(text, 
                            return_offsets_mapping=True,
                            max_length=MAX_SEQUENCE_LENGTH,
                            truncation=True,
                            add_special_tokens=True)
    
    roberta_tokens = roberta_encoding.tokens()
    roberta_token_ids = roberta_encoding.input_ids
    
    # Map spaCy tokens to RoBERTa subword indices
    mapping_spacy_to_roberta, processed_text = align_spacy_to_roberta(text, tokenizer, spacy_nlp)
    
    # Create RoBERTa-based dependency graph
    edges = []

    for (token, dep, head, token_idx, head_idx) in spacy_dependencies:
        if token_idx >= len(mapping_spacy_to_roberta) or head_idx >= len(mapping_spacy_to_roberta):
            continue  # Skip if indices are out of bounds
            
        token_roberta_indices = mapping_spacy_to_roberta[token_idx]
        head_roberta_indices = mapping_spacy_to_roberta[head_idx]

        for t_roberta_idx in token_roberta_indices:
            for h_roberta_idx in head_roberta_indices:
                # Ensure indices are within bounds
                if t_roberta_idx < len(roberta_token_ids) and h_roberta_idx < len(roberta_token_ids):
                    edges.append([t_roberta_idx, h_roberta_idx])
    
    # Construct adjacency matrix for the graph
    ed = []
    ed_weights = []
    actual_num_nodes = len(roberta_token_ids)
    
    # Ensure we don't exceed the maximum sequence length
    max_nodes = min(actual_num_nodes, MAX_SEQUENCE_LENGTH)
    
    for i in range(max_nodes):
        for j in range(max_nodes):
            if ([i, j] in edges or i == j) and i < max_nodes and j < max_nodes:
                ed.append([i, j])
                ed_weights.append(1.0)

    # Convert edges to torch tensors
    if ed:  # Check if there are any edges
        edge_index = torch.tensor(ed, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(ed_weights, dtype=torch.float)
    else:
        # Create empty edge tensors if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float)
    
    # Final validation: ensure all edge indices are within bounds
    if edge_index.numel() > 0:
        valid_edge_mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
        edge_index = edge_index[:, valid_edge_mask]
        edge_weight = edge_weight[valid_edge_mask]
    
    # Construct the graph data object for torch-geometric
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=max_nodes)
    
    return graph_data, roberta_token_ids

def load_articles(obj):
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))

    restyle_dict = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_A.pkl', 'rb'))

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    x_test_res = restyle_dict['news']

    # Use consistent text length and ensure it's within limits
    x_train_graph = load_graphs('graph/newstrain_t_all_'+obj+'.Robertagraph', x_train)
    x_test_graph = load_graphs('graph/newstest_t_all_'+obj+'.Robertagraph', x_test)
    x_test_res_graph = load_graphs('graph/newstest_t_all_res_a_'+obj+'.Robertagraph', x_test_res)
    
    return x_train, x_test, x_test_res, y_train, y_test, x_train_graph, x_test_graph, x_test_res_graph

def load_articles_all(obj):
    print('Dataset: ', obj)
    print("loading Test news articles")

    restyle_dictB = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_B.pkl', 'rb'))
    restyle_dictC = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_C.pkl', 'rb'))
    restyle_dictD = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_D.pkl', 'rb'))

    x_test_resB = restyle_dictB['news']
    x_test_resC = restyle_dictC['news']
    x_test_resD = restyle_dictD['news']

    
    x_test_res_graphb = load_graphs('graph/newstest_t_all_res_b_'+obj+'.Robertagraph', x_test_resB)
    x_test_res_graphc = load_graphs('graph/newstest_t_all_res_c_'+obj+'.Robertagraph', x_test_resC)
    x_test_res_graphd = load_graphs('graph/newstest_t_all_res_d_'+obj+'.Robertagraph', x_test_resD)
    
    return x_test_resB, x_test_resC, x_test_resD, x_test_res_graphb, x_test_res_graphc, x_test_res_graphd

def file_exists(file_path):
    return os.path.exists(file_path)

def load_graphs(file_path, texts):
    """
    Load graphs with proper text truncation to avoid token length issues.
    """
    if file_exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    
    idx2graph = {}
    fout = open(file_path, 'wb')
    
    for i, text in enumerate(texts):
        if i % 100 == 0:  # Progress indicator
            print(f"Processing graph {i}/{len(texts)}")
        
        try:
            # Truncate text to ensure it fits within RoBERTa limits
            truncated_text = truncate_text_for_roberta(text)
            adj_matrix, _ = construct_text_graph(truncated_text)
            idx2graph[i] = adj_matrix
        except Exception as e:
            print(f"Error processing text {i}: {e}")
            # Create empty graph as fallback
            empty_graph = Data(edge_index=torch.empty((2, 0), dtype=torch.long), 
                             edge_weight=torch.empty(0, dtype=torch.float),
                             num_nodes=1)
            idx2graph[i] = empty_graph
    
    pickle.dump(idx2graph, fout)        
    fout.close()
    return pickle.load(open(file_path, 'rb'))

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
    
    x_train_res1_graph = load_graphs('graph/x_train_res1_all_'+obj+'.Robertagraph', x_train_res1)
    x_train_res1_2_graph = load_graphs('graph/x_train_res1_all_2_'+obj+'.Robertagraph', x_train_res1_2)
    x_train_res2_graph = load_graphs('graph/x_train_res2_all_'+obj+'.Robertagraph', x_train_res2)
    x_train_res2_2_graph = load_graphs('graph/x_train_res2_all_2_'+obj+'.Robertagraph', x_train_res2_2)

    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]   
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]  
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]
    replace_idx = replace_idx.tolist()

    # Access and replace correctly
    for idx in replace_idx:
        x_train_res1_graph[idx] = x_train_res1_2_graph[idx]
        x_train_res2_graph[idx] = x_train_res2_2_graph[idx]

    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t, x_train_res1_graph, x_train_res2_graph
    
# Keep the existing edge transformation functions unchanged
def edge_to_node_graph(input_graph):
    """
    Transforms a graph where edges become nodes in the output graph.
    """
    edge_index = input_graph.edge_index
    edge_weight = input_graph.edge_weight if hasattr(input_graph, 'edge_weight') else None
    
    num_edges = edge_index.shape[1]
    
    edge_to_idx = {}
    edges_list = []
    
    for i in range(num_edges):
        edge = (edge_index[0, i].item(), edge_index[1, i].item())
        edge_to_idx[edge] = i
        edges_list.append(edge)
    
    new_edges = []
    new_edge_weights = []
    
    for i in range(num_edges):
        edge1 = edges_list[i]
        src1, dst1 = edge1
        
        for j in range(i + 1, num_edges):
            edge2 = edges_list[j]
            src2, dst2 = edge2
            
            if src1 == src2 or src1 == dst2 or dst1 == src2 or dst1 == dst2:
                new_edges.append([i, j])
                new_edges.append([j, i])
                
                if edge_weight is not None:
                    weight = (edge_weight[i] + edge_weight[j]) / 2.0
                    new_edge_weights.extend([weight, weight])
                else:
                    new_edge_weights.extend([1.0, 1.0])
    
    if new_edges:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_weight = torch.tensor(new_edge_weights, dtype=torch.float)
    else:
        new_edge_index = torch.tensor([[], []], dtype=torch.long)
        new_edge_weight = torch.tensor([], dtype=torch.float)
    
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

def edge_to_node_graph_directed(input_graph, consider_direction=True):
    """
    Advanced version that considers edge direction when determining neighbors.
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
                if dst1 == src2:
                    new_edges.append([i, j])
                    
                    if edge_weight is not None:
                        weight = edge_weight[i] * edge_weight[j]
                        new_edge_weights.append(weight)
                    else:
                        new_edge_weights.append(1.0)
            else:
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
    """
    transformed_graphs = {}
    
    for idx, graph in graphs_dict.items():
        try:
            transformed_graph = edge_to_node_graph(graph)
            transformed_graphs[idx] = transformed_graph
        except Exception as e:
            print(f"Error transforming graph at index {idx}: {e}")
            transformed_graphs[idx] = graph
    
    return transformed_graphs

# Test function to verify alignment correctness
def test_alignment(text_sample="The quick brown fox jumps over the lazy dog.", debug=True):
    """
    Test the spaCy-RoBERTa alignment with a sample text.
    """
    print("=" * 50)
    print("TESTING SPACY-ROBERTA ALIGNMENT")
    print("=" * 50)
    
    mapping, processed_text = align_spacy_to_roberta(text_sample, debug=debug)
    
    print("\nAlignment Test Results:")
    print(f"Original text: '{text_sample}'")
    print(f"Processed text: '{processed_text}'")
    
    # Process with spaCy
    doc = nlp(processed_text)
    
    # Tokenize with RoBERTa
    encoded = tokenizer(processed_text, return_offsets_mapping=True, add_special_tokens=True)
    roberta_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    
    print(f"\nspaCy tokens ({len(list(doc))}):")
    for i, token in enumerate(doc):
        roberta_indices = mapping[i] if i < len(mapping) else []
        roberta_subtokens = [roberta_tokens[idx] for idx in roberta_indices if idx < len(roberta_tokens)]
        print(f"  {i}: '{token.text}' [{token.idx}:{token.idx + len(token.text)}] -> {roberta_indices} {roberta_subtokens}")
    
    print(f"\nRoBERTa tokens ({len(roberta_tokens)}):")
    for i, (token, offset) in enumerate(zip(roberta_tokens, encoded["offset_mapping"])):
        print(f"  {i}: '{token}' {offset}")
    
    # Verify no spaCy token is unmapped
    unmapped_count = sum(1 for indices in mapping if not indices)
    print(f"\nUnmapped spaCy tokens: {unmapped_count}")
    
    return mapping, processed_text

# Example usage and additional utility functions
def validate_alignment_quality(text_samples):
    """
    Validate alignment quality across multiple text samples.
    """
    total_unmapped = 0
    total_spacy_tokens = 0
    
    for i, text in enumerate(text_samples[:5]):  # Test first 5 samples
        print(f"\nTesting sample {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        mapping, _ = align_spacy_to_roberta(text, debug=False)
        
        unmapped = sum(1 for indices in mapping if not indices)
        total_unmapped += unmapped
        total_spacy_tokens += len(mapping)
        
        print(f"  Unmapped tokens: {unmapped}/{len(mapping)} ({100*unmapped/len(mapping):.1f}%)")
    
    print(f"\nOverall alignment quality:")
    print(f"Total unmapped: {total_unmapped}/{total_spacy_tokens} ({100*total_unmapped/total_spacy_tokens:.1f}%)")
    
    return total_unmapped / total_spacy_tokens if total_spacy_tokens > 0 else 0

def transform_loaded_graphs(file_path, texts):
    """
    Load graphs and apply edge-to-node transformation.
    """
    original_graphs = load_graphs(file_path, texts)
    
    transformed_file_path = file_path.replace('.Robertagraph', '_edge2node.Robertagraph')
    
    if file_exists(transformed_file_path):
        return pickle.load(open(transformed_file_path, 'rb'))
    
    transformed_graphs = batch_edge_to_node_transform(original_graphs)
    
    with open(transformed_file_path, 'wb') as fout:
        pickle.dump(transformed_graphs, fout)
    
    return transformed_graphs