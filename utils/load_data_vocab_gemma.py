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
from transformers import AutoTokenizer, AutoModel
import torch
from torch_geometric.data import Data
import os
import math
import nltk
from collections import defaultdict, Counter
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
nltk.download('punkt')

# Load Gemma tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')

# Gemma 2 2B has 2304 hidden dimensions
GEMMA_HIDDEN_SIZE = 2304

def tokenize_gemma(texts, tokenizer):
    tokenized_docs = []
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        tokenized_docs.append(tokens)
    return tokenized_docs

def build_vocab_and_cooc(tokenized_docs, window_size=10, min_count=0):
    vocab_counter = Counter()
    co_occurrence = defaultdict(int)
    total_windows = 0

    for tokens in tokenized_docs:
        for i in range(len(tokens)):
            window = tokens[i:i+window_size]
            unique = set(window)
            total_windows += 1

            for token in unique:
                vocab_counter[token] += 1
            for t1, t2 in combinations(sorted(unique), 2):
                co_occurrence[(t1, t2)] += 1

    vocab = {w for w, c in vocab_counter.items() if c >= min_count}
    token2idx = {token: idx for idx, token in enumerate(sorted(vocab))}

    return vocab, token2idx, vocab_counter, co_occurrence, total_windows

def npmi(w1, w2, vocab_counter, co_occurrence, total_windows):
    p_x = vocab_counter[w1] / total_windows
    p_y = vocab_counter[w2] / total_windows
    p_xy = co_occurrence[(w1, w2)] / total_windows
    if p_xy == 0:
        return None
    pmi = math.log(p_xy / (p_x * p_y) + 1e-10)
    return pmi / (-math.log(p_xy + 1e-10))

def compute_npmi_edges(token2idx, vocab_counter, co_occurrence, total_windows):
    edges = []
    weights = []

    for (w1, w2), _ in co_occurrence.items():
        if w1 in token2idx and w2 in token2idx:
            score = npmi(w1, w2, vocab_counter, co_occurrence, total_windows)
            if score and score > 0:
                edges.append((token2idx[w1], token2idx[w2]))
                weights.append(score)

    return edges, weights

def build_pyg_graph(texts, tokenizer=tokenizer, window_size=10, min_count=1):
    """Build PyG graph with Gemma-compatible node features"""
    tokenized = tokenize_gemma(texts, tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(tokenized, window_size, min_count)
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    
    print(f"Building graph: {len(token2idx)} tokens, {len(edges)} edges")
    
    if not edges:
        print("No edges found after filtering. Creating minimal graph.")
        # Create minimal graph with Gemma-sized features
        x = torch.randn(max(1, len(token2idx)), GEMMA_HIDDEN_SIZE)  # Gemma embedding size
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

    # PyG format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    edge_weight = torch.tensor(weights, dtype=torch.float)  # shape [num_edges]

    # Create Gemma-compatible node features
    x = torch.randn(len(token2idx), GEMMA_HIDDEN_SIZE)  # Gemma embedding dimension
    
    print(f"Created graph with node features shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

def build_pyg_graph_with_gemma_init(texts, tokenizer=tokenizer, window_size=10, min_count=1):
    """Build PyG graph and initialize with actual Gemma embeddings"""
    tokenized = tokenize_gemma(texts, tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(tokenized, window_size, min_count)
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    
    print(f"Building graph with Gemma initialization: {len(token2idx)} tokens, {len(edges)} edges")
    
    if not edges:
        print("No edges found, creating minimal graph")
        x = torch.randn(max(1, len(token2idx)), GEMMA_HIDDEN_SIZE)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

    # PyG format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Initialize node features with Gemma embeddings
    print("Initializing node features with Gemma embeddings...")
    gemma_model = AutoModel.from_pretrained('google/gemma-2-2b')
    gemma_model.eval()
    
    # Create node features matrix
    x = torch.zeros(len(token2idx), GEMMA_HIDDEN_SIZE)
    
    # Get Gemma embeddings for each token in vocabulary
    tokens = list(token2idx.keys())
    with torch.no_grad():
        for i, token in enumerate(tokens):
            # Create a simple input with just this token
            token_ids = tokenizer.encode(token, add_special_tokens=True, return_tensors='pt')
            outputs = gemma_model(token_ids)
            # Use mean pooling of all token embeddings as token representation
            token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            vocab_idx = token2idx[token]
            x[vocab_idx] = token_embedding
    
    print(f"Initialized {len(token2idx)} node features with Gemma embeddings")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

def build_pyg_graph_efficient(texts, tokenizer=tokenizer, window_size=10, min_count=1, 
                            node_feature_dim=GEMMA_HIDDEN_SIZE, use_gemma_init=False):
    """
    Efficient graph building with proper node feature dimensions for Gemma
    
    Args:
        texts: List of text documents
        tokenizer: Gemma tokenizer
        window_size: Window size for co-occurrence
        min_count: Minimum token frequency
        node_feature_dim: Dimension of node features (default 2304 for Gemma 2 2B)
        use_gemma_init: Whether to initialize with Gemma embeddings (slower but better)
    """
    tokenized = tokenize_gemma(texts, tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(tokenized, window_size, min_count)
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    
    num_tokens = len(token2idx)
    print(f"Building efficient graph: {num_tokens} tokens, {len(edges)} edges")
    print(f"Node feature dimension: {node_feature_dim}")
    
    if not edges:
        print("Warning: No edges found, creating graph with isolated nodes")
        # Still create a valid graph structure
        x = torch.randn(max(1, num_tokens), node_feature_dim) * 0.1  # Small random init
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

    # Create edge tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Create appropriately sized node features
    if use_gemma_init:
        print("Initializing with Gemma embeddings (this may take a while)...")
        x = initialize_with_gemma_embeddings(token2idx, tokenizer, node_feature_dim)
    else:
        # Simple random initialization (much faster)
        x = torch.randn(num_tokens, node_feature_dim) * 0.1
        print(f"Initialized with random features: {x.shape}")

    # Verify everything is consistent
    assert x.size(0) == num_tokens, f"Node features size {x.size(0)} != vocab size {num_tokens}"
    assert edge_index.max().item() < num_tokens, f"Edge index {edge_index.max().item()} >= vocab size {num_tokens}"
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

def initialize_with_gemma_embeddings(token2idx, tokenizer, embedding_dim=GEMMA_HIDDEN_SIZE):
    """Initialize node features with Gemma embeddings"""
    gemma_model = AutoModel.from_pretrained('google/gemma-2-2b')
    gemma_model.eval()
    
    num_tokens = len(token2idx)
    x = torch.zeros(num_tokens, embedding_dim)
    
    # Process tokens in batches for efficiency
    tokens = list(token2idx.keys())
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i+batch_size]
            
            for token in batch_tokens:
                try:
                    # Encode single token
                    token_ids = tokenizer.encode(token, add_special_tokens=True, return_tensors='pt')
                    outputs = gemma_model(token_ids)
                    
                    # Use average of all token embeddings
                    token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = token_embedding[:embedding_dim]  # Ensure correct size
                    
                except Exception as e:
                    print(f"Warning: Could not get Gemma embedding for token '{token}': {e}")
                    # Use random initialization for problematic tokens
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = torch.randn(embedding_dim) * 0.1
    
    return x

from torch_geometric.utils import subgraph

def get_vocab_subgraph(token_ids, vocab_graph: Data, token2idx: dict, tokenizer) -> Data:
    """
    Returns a subgraph of the vocab graph that includes only the tokens from token_ids.
    
    Args:
        token_ids (List[int]): Gemma token IDs (from tokenizer).
        vocab_graph (Data): PyTorch Geometric vocab graph.
        token2idx (dict): Token-to-node index map used in vocab graph.
        tokenizer: Gemma tokenizer instance.

    Returns:
        Data: Subgraph as PyG Data object.
    """
    # Convert IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Get node indices in the graph for those tokens
    node_indices = [token2idx[token] for token in tokens if token in token2idx]

    if not node_indices:
        raise ValueError("None of the input tokens are present in the vocab graph.")

    # Convert to torch tensor
    node_mask = torch.tensor(node_indices, dtype=torch.long)

    # Extract subgraph
    edge_index, edge_attr = subgraph(node_mask, vocab_graph.edge_index, edge_attr=vocab_graph.edge_attr, relabel_nodes=True)

    # Optional: keep only relevant rows from x
    x = vocab_graph.x[node_mask]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def get_vocab_subgraph_from_text(text: str, vocab_graph: Data, token2idx: dict, tokenizer) -> Data:
    """
    Returns a subgraph of the vocab graph containing only Gemma tokens from the input text.
    
    Args:
        text (str): Input sentence or paragraph.
        vocab_graph (Data): Full vocab graph.
        token2idx (dict): Mapping from token string to node index in the graph.
        tokenizer: Gemma tokenizer.

    Returns:
        Data: Subgraph as torch_geometric.data.Data
    """
    # Tokenize the input text
    tokens = tokenizer.tokenize(text.lower())
    
    # Map to graph node indices
    node_indices = [token2idx[token] for token in tokens if token in token2idx]
    
    if not node_indices:
        raise ValueError("None of the tokens from the input text are in the vocabulary graph.")

    node_mask = torch.tensor(node_indices, dtype=torch.long)

    # Extract the subgraph
    edge_index, edge_attr = subgraph(node_mask, vocab_graph.edge_index, edge_attr=vocab_graph.edge_attr, relabel_nodes=True)

    # Get node features
    x = vocab_graph.x[node_mask]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def split_news_by_binary_label(train_dict):
    """
    Splits news texts into two lists based on binary labels (0 or 1).

    Args:
        train_dict (dict): Contains 'news' and 'labels'.

    Returns:
        Tuple[List[str], List[str]]: (news_0, news_1)
    """
    news = train_dict['news']
    labels = train_dict['labels']

    if len(news) != len(labels):
        raise ValueError("The number of news items must match the number of labels.")

    news_0 = [n for n, l in zip(news, labels) if l == 0]
    news_1 = [n for n, l in zip(news, labels) if l == 1]

    return news_0, news_1

def visualize_vocab_subgraph(text, vocab_graph, token2idx, tokenizer=tokenizer, top_n=20):
    # Get the subgraph
    subgraph = get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer)
    
    # Convert to NetworkX graph
    G = to_networkx(subgraph, to_undirected=True)

    # Get node ID to token mapping (since node indices are relabeled in subgraph)
    tokens = tokenizer.tokenize(text.lower())
    present_tokens = [t for t in tokens if t in token2idx]
    idx_map = {i: tok for i, tok in enumerate(present_tokens)}

    # Draw graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)  # Increase k for more spacing

    # Keep only top_n edges by weight magnitude
    if subgraph.edge_attr is not None:
        edge_list = list(G.edges())
        edge_weights = subgraph.edge_attr.tolist()
        
        # Sort edges by absolute weight (highest first)
        edge_data = sorted(zip(edge_list, edge_weights), key=lambda x: abs(x[1]), reverse=True)
        
        # Keep only top_n edges (or all if fewer than top_n)
        top_edges = edge_data[:min(top_n, len(edge_data))]
        
        # Create filtered graph with only top edges
        G_filtered = nx.Graph()
        G_filtered.add_nodes_from(G.nodes())
        
        # Add only the top edges with their weights
        filtered_edges = []
        filtered_weights = []
        for (u, v), w in top_edges:
            G_filtered.add_edge(u, v, weight=w)
            filtered_edges.append((u, v))
            filtered_weights.append(w)
        
        # Draw nodes (all original nodes)
        nx.draw_networkx_nodes(G_filtered, pos, node_color='lightblue', node_size=1000, alpha=0.8)
        
        # Draw only top edges with width proportional to weight
        edge_widths = [max(0.5, 5 * abs(w)) for w in filtered_weights]
        edge_colors = ['green' if w > 0 else 'red' for w in filtered_weights]
        
        nx.draw_networkx_edges(G_filtered, pos, 
                               edgelist=filtered_edges,
                               width=edge_widths, 
                               edge_color=edge_colors, 
                               alpha=0.7)
        
        # Draw node labels
        nx.draw_networkx_labels(G_filtered, pos, labels=idx_map, font_size=12, font_weight='bold')
        
        # Draw edge weight labels for top edges
        edge_labels = {
            (u, v): f"{w:.3f}"  # 3 decimal places for more precision
            for (u, v), w in zip(filtered_edges, filtered_weights)
        }
        
        nx.draw_networkx_edge_labels(
            G_filtered, pos, 
            edge_labels=edge_labels, 
            font_size=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2),
            font_weight='bold'
        )
    else:
        # If no edge attributes, draw all edges
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, edge_color='gray', alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels=idx_map, font_size=12, font_weight='bold')

    plt.title(f"Vocabulary Subgraph (Top {min(top_n, len(G.edges()))} Edges by Weight)", fontsize=16)
    plt.axis("off")
    
    # Add a legend for edge colors
    if subgraph.edge_attr is not None:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=4, label='Positive correlation'),
            Line2D([0], [0], color='red', lw=4, label='Negative correlation')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print table of top edges and their weights
    if subgraph.edge_attr is not None:
        print(f"\nTop {min(top_n, len(top_edges))} edges by weight magnitude:")
        print(f"{'Token 1':<15} {'Token 2':<15} {'Weight':<10}")
        print("-" * 40)
        for (u, v), w in top_edges:
            print(f"{idx_map[u]:<15} {idx_map[v]:<15} {w:.4f}")

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
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')

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
    """
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)     )

    valid_mask = valid_mask.view(-1)
    return edge_index[:, valid_mask],edge_type[valid_mask]

def clean_edge_index2(edge_index,edge_type ,edge_batch=0, num_nodes=511):
    """
    Cleans the edge_index tensor by removing edges that reference nodes
    outside the allowed range for each batch.
    """
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)     )

    valid_mask = valid_mask.view(-1)
    return edge_index[:, valid_mask],edge_type[valid_mask]

def clean_edge_index1(edge_index,edge_type ,edge_batch=0, num_nodes=511):
    """
    Cleans the edge_index tensor by removing edges that reference nodes
    outside the allowed range for each batch.
    """
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices)  )

    valid_mask = valid_mask.view(-1)
    return edge_index[:, valid_mask],edge_type[valid_mask]

def clean_edge_index(edge_index, edge_batch=0, num_nodes=511):
    """
    Cleans the edge_index tensor by removing edges that reference nodes
    outside the allowed range for each batch.
    """
    min_node_indices = edge_batch * num_nodes
    max_node_indices = (edge_batch + 1) * num_nodes
    
    valid_mask = (
        (edge_index[0] >= min_node_indices) & (edge_index[0] < max_node_indices) &
        (edge_index[1] >= min_node_indices) & (edge_index[1] < max_node_indices) 
    )

    valid_mask = valid_mask.view(-1)
    return edge_index[:, valid_mask]

def map_spacy_to_gemma(text, tokenizer, spacy_nlp):
    """
    Maps spaCy tokens to Gemma subword indices using character-based alignment.

    Args:
        text (str): Input text.
        tokenizer: Gemma tokenizer.
        spacy_nlp: spaCy NLP model.

    Returns:
        list: List of tuples (spacy_token_index, [gemma_subword_indices]).
    """

    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]

    # Tokenize with Gemma
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    gemma_tokens = encoded.tokens()
    offsets = encoded.offset_mapping  # Each subword's (start, end) character positions

    # Map each character position to its corresponding Gemma token index
    char_to_gemma = {}
    for gemma_idx, (start, end) in enumerate(offsets):
        for char_idx in range(start, end):  
            char_to_gemma[char_idx] = gemma_idx

    # Map spaCy tokens to corresponding Gemma subwords
    mapping_spacy_to_gemma = []
    for spacy_idx, token in enumerate(doc):
        token_start = token.idx  # Start position in text
        token_end = token_start + len(token)

        # Collect all Gemma subword indices that align with this token
        subword_indices = sorted(set(char_to_gemma.get(i) for i in range(token_start, token_end) if i in char_to_gemma))

        mapping_spacy_to_gemma.append((spacy_idx, subword_indices))

    return mapping_spacy_to_gemma

def construct_text_graph(text, save_path=None, tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Constructs a graph representation of a text using Gemma tokens and spaCy dependencies.

    Args:
        text (str): Input text.
        tokenizer: Gemma tokenizer.
        spacy_nlp: spaCy NLP model.
        save_path (str): File path to save the graph information.

    Returns:
        torch_geometric.data.Data: Graph representation with edge index.
        list: List of token IDs for Gemma tokens.
        pd.DataFrame: DataFrame containing graph edges and their dependency relations.
    """
    global graph_counter

    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_dependencies = [(token.i, token.dep_, token.head.i) for token in doc]  # (token_idx, dep, head_idx)

    # Tokenize with Gemma
    gemma_tokens = tokenizer.tokenize(text)
    gemma_token_ids = tokenizer.convert_tokens_to_ids(gemma_tokens)

    # Get mapping from spaCy tokens to Gemma subwords
    mapping_spacy_to_gemma = map_spacy_to_gemma(text, tokenizer, spacy_nlp)

    # Create Gemma-based dependency graph
    edges = []
    edge_weights=[]
    dependency_relations = []
    
    for token_idx, dep, head_idx in spacy_dependencies:
        token_gemma_indices = [gemma_idx for _, gemma_indices in mapping_spacy_to_gemma if _ == token_idx for gemma_idx in gemma_indices]
        head_gemma_indices = [gemma_idx for _, gemma_indices in mapping_spacy_to_gemma if _ == head_idx for gemma_idx in gemma_indices]

        for t_gemma_idx in token_gemma_indices:
            for h_gemma_idx in head_gemma_indices:
                edges.append([t_gemma_idx, h_gemma_idx])
                dependency_relations.append(DEP_RELATION_TYPE.get(dep, DEP_RELATION_TYPE["other"]))                              
                edge_weights.append(DEP_RELATION_WEIGHTS.get(dep, DEP_RELATION_WEIGHTS["other"]))
                
    # Convert edges to torch tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    edge_type = torch.tensor(dependency_relations, dtype=torch.long)

    # Construct the graph data object for torch-geometric
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight, edge_type=edge_type)
    df=[]

    graph_counter += 1  # Increment graph ID

    return graph_data, gemma_token_ids, df

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

    news_o,news_1=split_news_by_binary_label(train_dict)
    
    x_trian_graphr=load_vocab_graphs('gr/vocab_gemma_r_'+obj+'.Rograph',news_o)
    x_trian_graphf=load_vocab_graphs('gr/vocab_gemma_f_'+obj+'.Rograph',news_1)

    return x_train, x_test, x_test_res, y_train, y_test,x_trian_graphr,x_trian_graphf

def load_vocab_graphs(file_path, texts, force_rebuild=False):
    """Load or build vocabulary graphs with correct Gemma-sized features"""
    if file_exists(file_path) and not force_rebuild:
        try:
            print(f"Loading existing vocabulary graph from {file_path}")
            vocab_data = pickle.load(open(file_path, 'rb'))
            
            # Validate the data structure
            if isinstance(vocab_data, dict) and 0 in vocab_data and 1 in vocab_data:
                graph, token2idx = vocab_data[0], vocab_data[1]
                
                # CRITICAL CHECK: Verify node feature dimensions
                if graph.x is not None:
                    node_feature_dim = graph.x.size(1)
                    num_nodes = graph.x.size(0)
                    vocab_size = len(token2idx)
                    
                    print(f"Loaded graph: {num_nodes} nodes, {vocab_size} vocab size, {node_feature_dim}D features")
                    
                    # Check if we have the old problematic format
                    if node_feature_dim == vocab_size:
                        print("⚠️  Detected old identity matrix format! Forcing rebuild...")
                        force_rebuild = True
                    elif node_feature_dim != GEMMA_HIDDEN_SIZE:
                        print(f"⚠️  Non-Gemma feature dimension ({node_feature_dim})! Forcing rebuild...")
                        force_rebuild = True
                    else:
                        print("✅ Graph format is compatible with Gemma")
                        return vocab_data
                else:
                    print("⚠️  Graph has no node features! Forcing rebuild...")
                    force_rebuild = True
            else:
                print("⚠️  Invalid graph format! Forcing rebuild...")
                force_rebuild = True
                
        except Exception as e:
            print(f"Error loading graph: {e}. Forcing rebuild...")
            force_rebuild = True
    
    if force_rebuild or not file_exists(file_path):
        print(f"Building new vocabulary graph for {len(texts)} texts...")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Build the graph with correct dimensions
        graph, token2idx = build_pyg_graph_efficient(
            texts, 
            tokenizer=tokenizer, 
            window_size=10, 
            min_count=1,
            node_feature_dim=GEMMA_HIDDEN_SIZE,  # Gemma dimension
            use_gemma_init=False    # Set to True for better initialization (slower)
        )
        
        # Save the new graph
        vocab_graph = {
            0: graph,
            1: token2idx,
            'metadata': {
                'num_texts': len(texts),
                'num_tokens': len(token2idx),
                'node_feature_dim': GEMMA_HIDDEN_SIZE,
                'num_edges': graph.edge_index.size(1) if graph.edge_index is not None else 0
            }
        }
        
        with open(file_path, 'wb') as fout:
            pickle.dump(vocab_graph, fout)
        
        print(f"✅ Built and saved new graph: {len(token2idx)} tokens, {GEMMA_HIDDEN_SIZE}D features")
        return vocab_graph

def build_pyg_graph_robust(texts, tokenizer=None, window_size=10, min_count=1):
    """
    Robust version of build_pyg_graph with better error handling for Gemma.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    
    try:
        # Tokenize texts
        print("Tokenizing texts...")
        tokenized = tokenize_gemma(texts, tokenizer)
        
        if not tokenized:
            raise ValueError("No texts were successfully tokenized")
        
        # Build vocabulary and co-occurrence
        print("Building vocabulary and co-occurrence matrix...")
        vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(
            tokenized, window_size, min_count
        )
        
        if not vocab:
            raise ValueError(f"No vocabulary found with min_count={min_count}")
        
        print(f"Vocabulary size: {len(vocab)}, Total windows: {total_windows}")
        
        # Compute edges
        print("Computing NPMI edges...")
        edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
        
        if not edges:
            print(f"Warning: No edges found with current parameters, lowering min_count to 0")
            # Try with lower min_count
            vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(
                tokenized, window_size, min_count=0
            )
            edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
            
            if not edges:
                # Create a minimal graph with just identity features
                print("Creating minimal graph with identity features")
                x = torch.eye(len(token2idx)) if len(token2idx) > 0 else torch.zeros(1, 1)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_weight = torch.zeros(0, dtype=torch.float)
                return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx
        
        # Create PyG graph
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
        
        # Create node features (identity matrix as placeholder for Gemma embeddings)
        x = torch.eye(len(token2idx))
        
        print(f"Created graph with {len(token2idx)} nodes and {len(edges)} edges")
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx
        
    except Exception as e:
        print(f"Error in build_pyg_graph_robust: {e}")
        # Return minimal valid graph
        x = torch.zeros(1, 1)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), {}

def get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer):
    """
    Robust version of get_vocab_subgraph_from_text with better error handling for Gemma.
    """
    try:
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        if not token2idx:
            raise ValueError("Empty token2idx vocabulary")
        
        # Tokenize the input text
        tokens = tokenizer.tokenize(text.lower())
        
        if not tokens:
            raise ValueError("No tokens found in text")
        
        # Map to graph node indices - remove duplicates while preserving order
        seen = set()
        node_indices = []
        for token in tokens:
            if token in token2idx and token not in seen:
                node_indices.append(token2idx[token])
                seen.add(token)
        
        if not node_indices:
            raise ValueError("None of the tokens from the input text are in the vocabulary graph")

        node_mask = torch.tensor(node_indices, dtype=torch.long)

        # Extract the subgraph
        from torch_geometric.utils import subgraph
        edge_index, edge_attr = subgraph(
            node_mask, 
            vocab_graph.edge_index, 
            edge_attr=vocab_graph.edge_attr, 
            relabel_nodes=True
        )

        # Get node features
        x = vocab_graph.x[node_mask]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    except Exception as e:
        print(f"Warning: Error in subgraph extraction: {e}")
        # Return minimal valid subgraph
        device = vocab_graph.x.device if hasattr(vocab_graph, 'x') and vocab_graph.x is not None else torch.device('cpu')
        return Data(
            x=torch.zeros(1, GEMMA_HIDDEN_SIZE).to(device),
            edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
            edge_attr=torch.zeros(0).to(device)
        )

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
    
    x_train_res1_graph=load_graphs('gr/nngemmax_train_res1'+obj+'.Rograph',x_train_res1)
    x_train_res1_2_graph=load_graphs('gr/nngemmax_train_res1_2'+obj+'.Rograph',x_train_res1_2)
    x_train_res2_graph=load_graphs('gr/nngemmax_train_res2'+obj+'.Rograph',x_train_res2)
    x_train_res2_2_graph=load_graphs('gr/nngemmax_train_res2_2'+obj+'.Rograph',x_train_res2_2)

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