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
import math
import nltk
from collections import defaultdict, Counter
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
nltk.download('punkt')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from collections import defaultdict
import seaborn as sns
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_bert(texts,tokenizer):
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
def npmi(w1, w2,vocab_counter,co_occurrence,total_windows):
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
            score = npmi(w1, w2,vocab_counter, co_occurrence, total_windows)
            if score and score > 0:
                edges.append((token2idx[w1], token2idx[w2]))
                weights.append(score)

    return edges, weights

def build_pyg_graph(texts,tokenizer=tokenizer, window_size=10, min_count=1,):
    tokenized = tokenize_bert(texts,tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(tokenized, window_size, min_count)
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    print(edges,total_windows,token2idx, counter, cooc, total_windows)
    if not edges:
        raise ValueError("No edges found after filtering. Try lowering min_count.")

    # PyG format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    edge_weight = torch.tensor(weights, dtype=torch.float)  # shape [num_edges]

    # Optional: create dummy node features (e.g., identity)
    x = torch.eye(len(token2idx))  # shape [num_nodes, num_nodes] (can be replaced with BERT embeddings if needed)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

from torch_geometric.utils import subgraph

def get_vocab_subgraph(token_ids, vocab_graph: Data, token2idx: dict, tokenizer) -> Data:
    """
    Returns a subgraph of the vocab graph that includes only the tokens from token_ids.
    
    Args:
        token_ids (List[int]): BERT token IDs (from tokenizer).
        vocab_graph (Data): PyTorch Geometric vocab graph.
        token2idx (dict): Token-to-node index map used in vocab graph.
        tokenizer: BERT tokenizer instance.

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
    Returns a subgraph of the vocab graph containing only BERT tokens from the input text.
    
    Args:
        text (str): Input sentence or paragraph.
        vocab_graph (Data): Full vocab graph.
        token2idx (dict): Mapping from token string to node index in the graph.
        tokenizer: BERT tokenizer.

    Returns:
        Data: Subgraph as torch_geometric.data.Data
    """
    # Tokenize the input text (ignore special tokens like [CLS], [SEP])
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


# def visualize_vocab_subgraph(text, vocab_graph, token2idx, tokenizer=tokenizer):
#     # Get the subgraph
#     subgraph = get_vocab_subgraph_from_text(text, vocab_graph, token2idx, tokenizer)
    
#     # Convert to NetworkX graph
#     G = to_networkx(subgraph, to_undirected=True)

#     # Get node ID to token mapping (since node indices are relabeled in subgraph)
#     tokens = tokenizer.tokenize(text.lower())
#     present_tokens = [t for t in tokens if t in token2idx]
#     idx_map = {i: tok for i, tok in enumerate(present_tokens)}

#     # Draw graph
#     plt.figure(figsize=(8, 6))
#     pos = nx.spring_layout(G, seed=42)

#     # Draw nodes and edges
#     nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
#     nx.draw_networkx_edges(G, pos, edge_color='gray')
#     nx.draw_networkx_labels(G, pos, labels=idx_map, font_size=12)

#     # Optional: draw edge weights (NPMI)
#     if subgraph.edge_attr is not None:
#         edge_labels = {
#             (u, v): f"{w:.2f}"
#             for (u, v), w in zip(G.edges(), subgraph.edge_attr.tolist())
#         }
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

#     plt.title("Vocabulary Subgraph")
#     plt.axis("off")
#     plt.show()
# def visualize_vocab_subgraph(text, vocab_graph, token2idx, tokenizer=tokenizer):
#     # Get the subgraph
#     subgraph = get_vocab_subgraph_from_text(text, vocab_graph, token2idx, tokenizer)
    
#     # Convert to NetworkX graph
#     G = to_networkx(subgraph, to_undirected=True)

#     # Get node ID to token mapping (since node indices are relabeled in subgraph)
#     tokens = tokenizer.tokenize(text.lower())
#     present_tokens = [t for t in tokens if t in token2idx]
#     idx_map = {i: tok for i, tok in enumerate(present_tokens)}

#     # Draw graph
#     plt.figure(figsize=(10, 8))  # Larger figure for better readability
#     pos = nx.spring_layout(G, seed=42, k=1.5)  # Increase k for more spacing

#     # Draw nodes and edges with varying edge widths based on weights
#     edge_weights = []
#     if subgraph.edge_attr is not None:
#         edge_weights = subgraph.edge_attr.tolist()
#         # Scale weights for visualization (multiply by 5 for better visibility)
#         edge_widths = [max(0.5, 5 * abs(w)) for w in edge_weights]
        
#         # Optional: Color edges based on weight value
#         edge_colors = ['green' if w > 0 else 'red' for w in edge_weights]
#     else:
#         edge_widths = [1.0] * len(G.edges())
#         edge_colors = ['gray'] * len(G.edges())
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, alpha=0.8)
    
#     # Draw edges with varying width and color
#     nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    
#     # Draw node labels
#     nx.draw_networkx_labels(G, pos, labels=idx_map, font_size=12, font_weight='bold')

#     # Draw edge weights with improved formatting and background
#     if subgraph.edge_attr is not None:
#         edge_labels = {
#             (u, v): f"{w:.3f}"  # 3 decimal places for more precision
#             for (u, v), w in zip(G.edges(), edge_weights)
#         }
#         # Add a white background to edge labels for better readability
#         nx.draw_networkx_edge_labels(
#             G, pos, 
#             edge_labels=edge_labels, 
#             font_size=10,
#             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2),
#             font_weight='bold'
#         )

#     plt.title("Vocabulary Subgraph with Edge Weights", fontsize=16)
#     plt.axis("off")
    
#     # Add a legend for edge colors
#     if subgraph.edge_attr is not None:
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], color='green', lw=4, label='Positive correlation'),
#             Line2D([0], [0], color='red', lw=4, label='Negative correlation')
#         ]
#         plt.legend(handles=legend_elements, loc='upper right')
    
#     plt.tight_layout()
#     plt.show()

def visualize_vocab_subgraph(text, vocab_graph, token2idx, tokenizer=tokenizer, top_n=20):
    # Get the subgraph
    subgraph = get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer)
    
    # Convert to NetworkX graph
    G = to_networkx(subgraph, to_undirected=True)

    # Get node ID to token mapping (since node indices are relabeled in subgraph)
    tokens = tokenizer.tokenize(text.lower())
    present_tokens = [t for t in tokens if t in token2idx]
    
    # Create a mapping from relabeled node indices to tokens
    # This is important because subgraph relabels nodes starting from 0
    idx_map = {i: tok for i, tok in enumerate(present_tokens)}

    # Draw graph
    plt.figure(figsize=(10, 8))
    
    # Keep only top_n edges by weight magnitude
    if subgraph.edge_attr is not None and len(G.edges()) > 0:
        edge_list = list(G.edges())
        edge_weights = subgraph.edge_attr.tolist()
        
        # Sort edges by absolute weight (highest first)
        edge_data = sorted(zip(edge_list, edge_weights), key=lambda x: abs(x[1]), reverse=True)
        
        # Keep only top_n edges (or all if fewer than top_n)
        top_edges = edge_data[:min(top_n, len(edge_data))]
        
        # Create filtered graph with only top edges
        G_filtered = nx.Graph()
        
        # Add only the nodes that are part of the top edges
        nodes_in_top_edges = set()
        for (u, v), w in top_edges:
            nodes_in_top_edges.add(u)
            nodes_in_top_edges.add(v)
        
        G_filtered.add_nodes_from(nodes_in_top_edges)
        
        # Add only the top edges with their weights
        filtered_edges = []
        filtered_weights = []
        for (u, v), w in top_edges:
            G_filtered.add_edge(u, v, weight=w)
            filtered_edges.append((u, v))
            filtered_weights.append(w)
        
        # Calculate positions only for nodes in the filtered graph
        pos = nx.spring_layout(G_filtered, seed=42, k=1.5)
        
        # Create filtered idx_map only for nodes present in filtered graph
        filtered_idx_map = {node: idx_map[node] for node in G_filtered.nodes() if node in idx_map}
        
        # Draw nodes (only nodes in filtered graph)
        nx.draw_networkx_nodes(G_filtered, pos, node_color='lightblue', node_size=1000, alpha=0.8)
        
        # Draw only top edges with width proportional to weight
        edge_widths = [max(0.5, 5 * abs(w)) for w in filtered_weights]
        edge_colors = ['green' if w > 0 else 'red' for w in filtered_weights]
        
        nx.draw_networkx_edges(G_filtered, pos, 
                               edgelist=filtered_edges,
                               width=edge_widths, 
                               edge_color=edge_colors, 
                               alpha=0.7)
        
        # Draw node labels for filtered nodes only
        nx.draw_networkx_labels(G_filtered, pos, labels=filtered_idx_map, font_size=12, font_weight='bold')
        
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
        
        # Update the title and table to reflect filtered graph
        plt.title(f"Vocabulary Subgraph (Top {len(filtered_edges)} Edges by Weight)", fontsize=16)
        
        # Print table of top edges and their weights
        print(f"\nTop {len(top_edges)} edges by weight magnitude:")
        print(f"{'Token 1':<15} {'Token 2':<15} {'Weight':<10}")
        print("-" * 40)
        for (u, v), w in top_edges:
            if u in filtered_idx_map and v in filtered_idx_map:
                print(f"{filtered_idx_map[u]:<15} {filtered_idx_map[v]:<15} {w:.4f}")
        
    else:
        # If no edge attributes or no edges, draw all nodes
        pos = nx.spring_layout(G, seed=42, k=1.5)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.0, edge_color='gray', alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels=idx_map, font_size=12, font_weight='bold')
        plt.title("Vocabulary Subgraph (No Edge Weights)", fontsize=16)

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased")
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

    news_o,news_1=split_news_by_binary_label(train_dict)
    # idx2graph = {}
    # fout = open('test_res'+obj+'.Rograph', 'wb')
    # for i,ele in enumerate(x_test_res):
    #     #print(i)
    #     adj_matrix,_ = construct_text_graph(ele[:600])
    #     idx2graph[i] = adj_matrix
    # pickle.dump(idx2graph, fout)        
    # fout.close()
    x_trian_graphr=load_vocab_graphs('gr/vocab_r_'+obj+'.Rograph',news_o)
    x_trian_graphf=load_vocab_graphs('gr/vocab_f_'+obj+'.Rograph',news_1)

    # x_test_graph=load_graphs('gr/nnberttest'+obj+'.Rograph',x_test)
    # x_test_res_graph=load_graphs('gr/nnberttest_res__a1'+obj+'.Rograph',x_test_res)
    return x_train, x_test, x_test_res, y_train, y_test,x_trian_graphr,x_trian_graphf

def load_vocab_graphs(file_path, texts, tokenizer=None, window_size=10, min_count=1, force_rebuild=False):
    """
    Load or build vocabulary graphs with better error handling and flexibility.
    
    Args:
        file_path (str): Path to save/load the vocabulary graph
        texts (list): List of text documents to build vocabulary from
        tokenizer: BERT tokenizer (defaults to global tokenizer)
        window_size (int): Window size for co-occurrence calculation
        min_count (int): Minimum token frequency threshold
        force_rebuild (bool): Whether to force rebuilding even if file exists
    
    Returns:
        dict: Dictionary containing graph and token2idx mapping
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Check if file exists and we don't want to force rebuild
    if file_exists(file_path) and not force_rebuild:
        try:
            print(f"Loading existing vocabulary graph from {file_path}")
            vocab_data = pickle.load(open(file_path, 'rb'))
            
            # Validate the loaded data structure
            if isinstance(vocab_data, dict) and 0 in vocab_data and 1 in vocab_data:
                graph, token2idx = vocab_data[0], vocab_data[1]
                print(f"Loaded vocabulary graph with {len(token2idx)} tokens")
                return vocab_data
            else:
                print("Invalid vocabulary graph format, rebuilding...")
        except Exception as e:
            print(f"Error loading vocabulary graph: {e}, rebuilding...")
    
    # Build new vocabulary graph
    print(f"Building vocabulary graph from {len(texts)} texts...")
    print(f"Parameters: window_size={window_size}, min_count={min_count}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Build the graph
        graph, token2idx = build_pyg_graph_robust(
            texts, 
            tokenizer=tokenizer, 
            window_size=window_size, 
            min_count=min_count
        )
        
        # Prepare data structure
        vocab_graph = {
            0: graph,
            1: token2idx,
            'metadata': {
                'num_texts': len(texts),
                'num_tokens': len(token2idx),
                'window_size': window_size,
                'min_count': min_count,
                'num_edges': graph.edge_index.size(1) if graph.edge_index is not None else 0
            }
        }
        
        # Save to file
        with open(file_path, 'wb') as fout:
            pickle.dump(vocab_graph, fout)
        
        print(f"Built and saved vocabulary graph:")
        print(f"  - Tokens: {len(token2idx)}")
        print(f"  - Edges: {graph.edge_index.size(1) if graph.edge_index is not None else 0}")
        print(f"  - Saved to: {file_path}")
        
        return vocab_graph
        
    except Exception as e:
        print(f"Error building vocabulary graph: {e}")
        # Return empty graph as fallback
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        empty_graph = Data(
            x=torch.zeros(1, 768).to(device),
            edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
            edge_attr=torch.zeros(0).to(device)
        )
        empty_vocab = {
            0: empty_graph,
            1: {},
            'metadata': {'error': str(e)}
        }
        return empty_vocab
def build_pyg_graph_robust(texts, tokenizer=None, window_size=10, min_count=1):
    """
    Robust version of build_pyg_graph with better error handling.
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    try:
        # Tokenize texts
        print("Tokenizing texts...")
        tokenized = tokenize_bert(texts, tokenizer)
        
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
        
        # Create node features (identity matrix as placeholder for BERT embeddings)
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
    Robust version of get_vocab_subgraph_from_text with better error handling.
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

        # Check if vocab_graph has the required attributes
        if not hasattr(vocab_graph, 'edge_index') or vocab_graph.edge_index is None:
            print("Warning: vocab_graph has no edge_index, creating minimal graph")
            # Create a minimal graph with just the nodes
            num_nodes = len(node_indices)
            x = torch.eye(num_nodes)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Extract the subgraph
        from torch_geometric.utils import subgraph
        edge_index, edge_attr = subgraph(
            node_mask, 
            vocab_graph.edge_index, 
            edge_attr=vocab_graph.edge_attr if hasattr(vocab_graph, 'edge_attr') else None, 
            relabel_nodes=True
        )

        # Get node features
        if hasattr(vocab_graph, 'x') and vocab_graph.x is not None:
            x = vocab_graph.x[node_mask]
        else:
            # Create identity matrix as fallback
            x = torch.eye(len(node_indices))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    except Exception as e:
        print(f"Warning: Error in subgraph extraction: {e}")
        # Return minimal valid subgraph with identity features
        num_nodes = 1
        try:
            # Try to use the actual number of tokens if possible
            tokens = tokenizer.tokenize(text.lower())
            valid_tokens = [t for t in tokens if t in token2idx]
            if valid_tokens:
                num_nodes = len(set(valid_tokens))
        except:
            pass
        
        return Data(
            x=torch.eye(num_nodes),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0)
        )
        
    except Exception as e:
        print(f"Warning: Error in subgraph extraction: {e}")
        # Return minimal valid subgraph
        device = vocab_graph.x.device if hasattr(vocab_graph, 'x') and vocab_graph.x is not None else torch.device('cpu')
        return Data(
            x=torch.zeros(1, 768).to(device),
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
    
def print_word_neighbors(word, vocab_graph, token2idx, top_k=10, min_weight=0.0):
    """
    Print the neighbors of a given word in the vocabulary graph.
    
    Args:
        word (str): The input word to find neighbors for
        vocab_graph (Data): PyTorch Geometric vocab graph
        token2idx (dict): Token-to-node index mapping
        top_k (int): Number of top neighbors to show (default: 10)
        min_weight (float): Minimum edge weight to consider (default: 0.0)
    """
    
    # Check if word exists in vocabulary
    if word not in token2idx:
        print(f"Word '{word}' not found in vocabulary.")
        # Try to find similar words
        similar_words = [token for token in token2idx.keys() if word in token or token in word]
        if similar_words:
            print(f"Similar words found: {similar_words[:10]}")
        return
    
    # Get the node index for the word
    word_idx = token2idx[word]
    
    # Create reverse mapping from index to token
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    # Find all edges connected to this word
    neighbors = []
    
    if hasattr(vocab_graph, 'edge_index') and vocab_graph.edge_index is not None:
        edge_index = vocab_graph.edge_index
        edge_attr = vocab_graph.edge_attr if hasattr(vocab_graph, 'edge_attr') else None
        
        # Find edges where the word is either source or target
        for i in range(edge_index.size(1)):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            weight = edge_attr[i].item() if edge_attr is not None else 1.0
            
            # Check if this edge involves our word
            if source_idx == word_idx:
                neighbor_idx = target_idx
                neighbor_token = idx2token.get(neighbor_idx, f"Unknown_{neighbor_idx}")
                neighbors.append((neighbor_token, weight, neighbor_idx))
            elif target_idx == word_idx:
                neighbor_idx = source_idx
                neighbor_token = idx2token.get(neighbor_idx, f"Unknown_{neighbor_idx}")
                neighbors.append((neighbor_token, weight, neighbor_idx))
    
    # Filter by minimum weight and remove duplicates
    unique_neighbors = {}
    for neighbor_token, weight, neighbor_idx in neighbors:
        if weight >= min_weight:
            if neighbor_token not in unique_neighbors or weight > unique_neighbors[neighbor_token][0]:
                unique_neighbors[neighbor_token] = (weight, neighbor_idx)
    
    # Sort by weight (descending) and get top_k
    sorted_neighbors = sorted(unique_neighbors.items(), key=lambda x: x[1][0], reverse=True)
    top_neighbors = sorted_neighbors[:top_k]
    
    # Print results
    print(f"\nNeighbors of '{word}' (showing top {len(top_neighbors)} out of {len(unique_neighbors)} total):")
    print("-" * 60)
    print(f"{'Neighbor':<20} {'Weight':<10} {'Node Index':<10}")
    print("-" * 60)
    
    if not top_neighbors:
        print("No neighbors found.")
    else:
        for neighbor_token, (weight, neighbor_idx) in top_neighbors:
            print(f"{neighbor_token:<20} {weight:<10.4f} {neighbor_idx:<10}")
    
    return top_neighbors


def print_word_neighbors_detailed(word, vocab_graph, token2idx, top_k=10, min_weight=0.0):
    """
    Print detailed information about word neighbors including statistics.
    
    Args:
        word (str): The input word to find neighbors for
        vocab_graph (Data): PyTorch Geometric vocab graph
        token2idx (dict): Token-to-node index mapping
        top_k (int): Number of top neighbors to show (default: 10)
        min_weight (float): Minimum edge weight to consider (default: 0.0)
    """
    
    # Check if word exists in vocabulary
    if word not in token2idx:
        print(f"Word '{word}' not found in vocabulary.")
        # Try to find similar words (case-insensitive partial matching)
        similar_words = []
        word_lower = word.lower()
        for token in token2idx.keys():
            if (word_lower in token.lower() or token.lower() in word_lower or 
                word_lower.startswith(token.lower()) or token.lower().startswith(word_lower)):
                similar_words.append(token)
        
        if similar_words:
            print(f"Similar words found: {similar_words[:10]}")
        return None
    
    # Get the node index for the word
    word_idx = token2idx[word]
    
    # Create reverse mapping from index to token
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    # Find all edges connected to this word
    neighbors = []
    all_weights = []
    
    if hasattr(vocab_graph, 'edge_index') and vocab_graph.edge_index is not None:
        edge_index = vocab_graph.edge_index
        edge_attr = vocab_graph.edge_attr if hasattr(vocab_graph, 'edge_attr') else None
        
        # Find edges where the word is either source or target
        for i in range(edge_index.size(1)):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            weight = edge_attr[i].item() if edge_attr is not None else 1.0
            
            # Check if this edge involves our word
            if source_idx == word_idx:
                neighbor_idx = target_idx
                neighbor_token = idx2token.get(neighbor_idx, f"Unknown_{neighbor_idx}")
                neighbors.append((neighbor_token, weight, neighbor_idx))
                all_weights.append(weight)
            elif target_idx == word_idx:
                neighbor_idx = source_idx
                neighbor_token = idx2token.get(neighbor_idx, f"Unknown_{neighbor_idx}")
                neighbors.append((neighbor_token, weight, neighbor_idx))
                all_weights.append(weight)
    
    # Filter by minimum weight and remove duplicates
    unique_neighbors = {}
    for neighbor_token, weight, neighbor_idx in neighbors:
        if weight >= min_weight:
            if neighbor_token not in unique_neighbors or weight > unique_neighbors[neighbor_token][0]:
                unique_neighbors[neighbor_token] = (weight, neighbor_idx)
    
    # Sort by weight (descending) and get top_k
    sorted_neighbors = sorted(unique_neighbors.items(), key=lambda x: x[1][0], reverse=True)
    top_neighbors = sorted_neighbors[:top_k]
    
    # Print detailed results
    print(f"\n" + "="*70)
    print(f"DETAILED NEIGHBORS ANALYSIS FOR: '{word}'")
    print(f"="*70)
    print(f"Word Index: {word_idx}")
    print(f"Total Neighbors: {len(unique_neighbors)}")
    print(f"Showing Top: {len(top_neighbors)}")
    
    if all_weights:
        print(f"Weight Statistics:")
        print(f"  - Min Weight: {min(all_weights):.4f}")
        print(f"  - Max Weight: {max(all_weights):.4f}")
        print(f"  - Avg Weight: {sum(all_weights)/len(all_weights):.4f}")
    
    print("\n" + "-"*70)
    print(f"{'Rank':<5} {'Neighbor':<25} {'Weight':<12} {'Node Index':<10}")
    print("-"*70)
    
    if not top_neighbors:
        print("No neighbors found.")
    else:
        for rank, (neighbor_token, (weight, neighbor_idx)) in enumerate(top_neighbors, 1):
            print(f"{rank:<5} {neighbor_token:<25} {weight:<12.4f} {neighbor_idx:<10}")
    
    # Print weight distribution
    if len(unique_neighbors) > top_k:
        print(f"\n... and {len(unique_neighbors) - top_k} more neighbors with lower weights")
    
    return top_neighbors


def find_word_in_vocabulary(partial_word, token2idx, max_results=20):
    """
    Find words in vocabulary that match a partial word.
    
    Args:
        partial_word (str): Partial word to search for
        token2idx (dict): Token-to-node index mapping
        max_results (int): Maximum number of results to return
    
    Returns:
        list: List of matching words
    """
    partial_word_lower = partial_word.lower()
    matches = []
    
    for token in token2idx.keys():
        token_lower = token.lower()
        if (partial_word_lower in token_lower or 
            token_lower.startswith(partial_word_lower) or
            token_lower.endswith(partial_word_lower)):
            matches.append(token)
    
    matches.sort()
    return matches[:max_results]


def explore_word_connections(word, vocab_graph, token2idx, depth=2, max_per_level=5):
    """
    Explore word connections up to a certain depth (neighbors of neighbors).
    
    Args:
        word (str): Starting word
        vocab_graph (Data): PyTorch Geometric vocab graph
        token2idx (dict): Token-to-node index mapping
        depth (int): How many levels deep to explore
        max_per_level (int): Maximum neighbors to show per level
    """
    
    if word not in token2idx:
        print(f"Word '{word}' not found in vocabulary.")
        return
    
    print(f"\nExploring connections for '{word}' (depth={depth}):")
    print("="*60)
    
    visited = set()
    current_level = [word]
    
    for level in range(depth):
        print(f"\nLevel {level + 1}:")
        print("-" * 30)
        
        next_level = []
        for current_word in current_level:
            if current_word in visited:
                continue
            visited.add(current_word)
            
            # Get neighbors for current word
            neighbors = print_word_neighbors(current_word, vocab_graph, token2idx, 
                                           top_k=max_per_level, min_weight=0.0)
            
            if neighbors:
                neighbor_words = [neighbor[0] for neighbor in neighbors]
                next_level.extend(neighbor_words)
                print(f"  {current_word} -> {', '.join(neighbor_words)}")
            else:
                print(f"  {current_word} -> (no neighbors)")
        
        current_level = list(set(next_level))  # Remove duplicates
        if not current_level:
            print("No more connections found.")
            break
def print_word_neighbors_in_text(word, text, vocab_graph, token2idx, tokenizer=tokenizer, top_k=10, min_weight=0.0):
    """
    Find and print neighbors of a word within a text-specific subgraph.
    
    Args:
        word (str): The word to find neighbors for
        text (str): The text to create subgraph from
        vocab_graph (Data): Full vocabulary graph
        token2idx (dict): Token-to-index mapping for full vocabulary
        tokenizer: BERT tokenizer
        top_k (int): Maximum number of neighbors to show
        min_weight (float): Minimum edge weight to consider
    
    Returns:
        list: List of (neighbor_word, weight) tuples
    """
    
    print(f"Analyzing word '{word}' in text context...")
    print("="*60)
    
    # First, extract subgraph based on text
    try:
        subgraph = get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer)
        print(f"✓ Created subgraph with {subgraph.x.size(0)} nodes and {subgraph.edge_index.size(1)} edges")
    except Exception as e:
        print(f"✗ Error creating subgraph: {e}")
        return []
    
    # Get tokens from the text to create mapping
    tokens = tokenizer.tokenize(text.lower())
    present_tokens = [t for t in tokens if t in token2idx]
    
    if not present_tokens:
        print("✗ No tokens from text found in vocabulary")
        return []
    
    # Create mapping from subgraph node indices to tokens
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in present_tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    
    # Create bidirectional mapping for subgraph
    subgraph_idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    subgraph_token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    
    print(f"✓ Text contains {len(unique_tokens)} unique vocabulary tokens")
    print(f"Tokens in subgraph: {unique_tokens}")
    
    # Check if the target word is in the subgraph
    if word not in subgraph_token_to_idx:
        print(f"✗ Word '{word}' not found in the text subgraph")
        
        # Find similar words in the subgraph
        similar_words = [token for token in unique_tokens if word in token or token in word]
        if similar_words:
            print(f"Similar words in subgraph: {similar_words}")
        else:
            print("No similar words found in subgraph")
        return []
    
    # Get the subgraph node index for the word
    word_subgraph_idx = subgraph_token_to_idx[word]
    print(f"✓ Word '{word}' found at subgraph index {word_subgraph_idx}")
    
    # Find neighbors in the subgraph
    neighbors = []
    
    if hasattr(subgraph, 'edge_index') and subgraph.edge_index is not None:
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr if hasattr(subgraph, 'edge_attr') else None
        
        for i in range(edge_index.size(1)):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            weight = edge_attr[i].item() if edge_attr is not None else 1.0
            
            # Check if this edge involves our word
            if source_idx == word_subgraph_idx and target_idx in subgraph_idx_to_token:
                neighbor_token = subgraph_idx_to_token[target_idx]
                neighbors.append((neighbor_token, weight, target_idx))
            elif target_idx == word_subgraph_idx and source_idx in subgraph_idx_to_token:
                neighbor_token = subgraph_idx_to_token[source_idx]
                neighbors.append((neighbor_token, weight, source_idx))
    
    # Filter by minimum weight and remove duplicates
    unique_neighbors = {}
    for neighbor_token, weight, neighbor_idx in neighbors:
        if weight >= min_weight:
            if neighbor_token not in unique_neighbors or weight > unique_neighbors[neighbor_token][0]:
                unique_neighbors[neighbor_token] = (weight, neighbor_idx)
    
    # Sort by weight (descending) and get top_k
    sorted_neighbors = sorted(unique_neighbors.items(), key=lambda x: x[1][0], reverse=True)
    top_neighbors = sorted_neighbors[:top_k]
    
    # Print results
    print(f"\nNeighbors of '{word}' in text subgraph:")
    print("-" * 50)
    print(f"Total neighbors found: {len(unique_neighbors)}")
    print(f"Showing top {len(top_neighbors)} neighbors:")
    print("-" * 50)
    print(f"{'Neighbor':<20} {'Weight':<10} {'Subgraph Index':<15}")
    print("-" * 50)
    
    if not top_neighbors:
        print("No neighbors found in this text context.")
    else:
        for neighbor_token, (weight, neighbor_idx) in top_neighbors:
            print(f"{neighbor_token:<20} {weight:<10.4f} {neighbor_idx:<15}")
    
    return [(neighbor_token, weight) for neighbor_token, (weight, _) in top_neighbors]


def print_word_neighbors_in_text_detailed(word, text, vocab_graph, token2idx, tokenizer=tokenizer, top_k=10, min_weight=0.0):
    """
    Detailed analysis of word neighbors within a text-specific subgraph.
    
    Args:
        word (str): The word to find neighbors for
        text (str): The text to create subgraph from
        vocab_graph (Data): Full vocabulary graph
        token2idx (dict): Token-to-index mapping for full vocabulary
        tokenizer: BERT tokenizer
        top_k (int): Maximum number of neighbors to show
        min_weight (float): Minimum edge weight to consider
    
    Returns:
        dict: Detailed analysis results
    """
    
    print(f"\n{'='*80}")
    print(f"DETAILED SUBGRAPH ANALYSIS FOR: '{word}'")
    print(f"{'='*80}")
    
    # Show text preview
    text_preview = text[:200] + "..." if len(text) > 200 else text
    print(f"Text preview: {text_preview}")
    print(f"Text length: {len(text)} characters")
    
    # Extract subgraph
    try:
        subgraph = get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer)
        print(f"\nSubgraph Statistics:")
        print(f"  - Nodes: {subgraph.x.size(0)}")
        print(f"  - Edges: {subgraph.edge_index.size(1)}")
        print(f"  - Has edge weights: {hasattr(subgraph, 'edge_attr') and subgraph.edge_attr is not None}")
    except Exception as e:
        print(f"Error creating subgraph: {e}")
        return {}
    
    # Get tokens and create mappings
    tokens = tokenizer.tokenize(text.lower())
    present_tokens = [t for t in tokens if t in token2idx]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in present_tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    
    subgraph_idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
    subgraph_token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
    
    print(f"\nTokenization Results:")
    print(f"  - Total BERT tokens: {len(tokens)}")
    print(f"  - Tokens in vocabulary: {len(present_tokens)}")
    print(f"  - Unique tokens in subgraph: {len(unique_tokens)}")
    
    # Check if target word exists
    if word not in subgraph_token_to_idx:
        print(f"\n✗ Word '{word}' not found in subgraph")
        
        # Find similar words
        similar_words = [token for token in unique_tokens if word in token or token in word]
        if similar_words:
            print(f"Similar words available: {similar_words}")
            
            # Analyze the most similar word
            most_similar = similar_words[0]
            print(f"\nAnalyzing most similar word: '{most_similar}'")
            return print_word_neighbors_in_text_detailed(most_similar, text, vocab_graph, token2idx, tokenizer, top_k, min_weight)
        
        return {}
    
    word_subgraph_idx = subgraph_token_to_idx[word]
    print(f"\n✓ Word '{word}' found at subgraph index {word_subgraph_idx}")
    
    # Find all neighbors
    neighbors = []
    all_weights = []
    
    if hasattr(subgraph, 'edge_index') and subgraph.edge_index is not None:
        edge_index = subgraph.edge_index
        edge_attr = subgraph.edge_attr if hasattr(subgraph, 'edge_attr') else None
        
        for i in range(edge_index.size(1)):
            source_idx = edge_index[0, i].item()
            target_idx = edge_index[1, i].item()
            weight = edge_attr[i].item() if edge_attr is not None else 1.0
            
            if source_idx == word_subgraph_idx and target_idx in subgraph_idx_to_token:
                neighbor_token = subgraph_idx_to_token[target_idx]
                neighbors.append((neighbor_token, weight, target_idx))
                all_weights.append(weight)
            elif target_idx == word_subgraph_idx and source_idx in subgraph_idx_to_token:
                neighbor_token = subgraph_idx_to_token[source_idx]
                neighbors.append((neighbor_token, weight, source_idx))
                all_weights.append(weight)
    
    # Process neighbors
    unique_neighbors = {}
    for neighbor_token, weight, neighbor_idx in neighbors:
        if weight >= min_weight:
            if neighbor_token not in unique_neighbors or weight > unique_neighbors[neighbor_token][0]:
                unique_neighbors[neighbor_token] = (weight, neighbor_idx)
    
    sorted_neighbors = sorted(unique_neighbors.items(), key=lambda x: x[1][0], reverse=True)
    top_neighbors = sorted_neighbors[:top_k]
    
    # Print detailed results
    print(f"\nNeighbor Analysis:")
    print(f"  - Total neighbor connections: {len(neighbors)}")
    print(f"  - Unique neighbors: {len(unique_neighbors)}")
    print(f"  - Neighbors above threshold: {len([n for n in unique_neighbors.values() if n[0] >= min_weight])}")
    
    if all_weights:
        print(f"\nWeight Statistics:")
        print(f"  - Min: {min(all_weights):.4f}")
        print(f"  - Max: {max(all_weights):.4f}")
        print(f"  - Average: {sum(all_weights)/len(all_weights):.4f}")
    
    print(f"\n{'-'*80}")
    print(f"{'Rank':<5} {'Neighbor':<25} {'Weight':<12} {'Subgraph Index':<15}")
    print(f"{'-'*80}")
    
    if not top_neighbors:
        print("No neighbors found in this text context.")
    else:
        for rank, (neighbor_token, (weight, neighbor_idx)) in enumerate(top_neighbors, 1):
            print(f"{rank:<5} {neighbor_token:<25} {weight:<12.4f} {neighbor_idx:<15}")
    
    # Return comprehensive results
    return {
        'word': word,
        'text_length': len(text),
        'subgraph_nodes': subgraph.x.size(0),
        'subgraph_edges': subgraph.edge_index.size(1),
        'total_neighbors': len(unique_neighbors),
        'top_neighbors': [(neighbor_token, weight) for neighbor_token, (weight, _) in top_neighbors],
        'weight_stats': {
            'min': min(all_weights) if all_weights else 0,
            'max': max(all_weights) if all_weights else 0,
            'avg': sum(all_weights)/len(all_weights) if all_weights else 0
        },
        'tokens_in_subgraph': unique_tokens
    }


def compare_word_context_neighbors(word, text1, text2, vocab_graph, token2idx, tokenizer=tokenizer, top_k=5):
    """
    Compare how a word's neighbors differ between two different text contexts.
    
    Args:
        word (str): Word to analyze
        text1 (str): First text context
        text2 (str): Second text context
        vocab_graph (Data): Vocabulary graph
        token2idx (dict): Token-to-index mapping
        tokenizer: BERT tokenizer
        top_k (int): Number of top neighbors to compare
    """
    
    print(f"\n{'='*80}")
    print(f"COMPARING WORD '{word}' IN DIFFERENT CONTEXTS")
    print(f"{'='*80}")
    
    print(f"\nCONTEXT 1:")
    print(f"Text preview: {text1[:100]}...")
    neighbors1 = print_word_neighbors_in_text(word, text1, vocab_graph, token2idx, tokenizer, top_k)
    
    print(f"\nCONTEXT 2:")
    print(f"Text preview: {text2[:100]}...")
    neighbors2 = print_word_neighbors_in_text(word, text2, vocab_graph, token2idx, tokenizer, top_k)
    
    # Compare results
    if neighbors1 and neighbors2:
        words1 = set([n[0] for n in neighbors1])
        words2 = set([n[0] for n in neighbors2])
        
        common = words1.intersection(words2)
        unique1 = words1 - words2
        unique2 = words2 - words1
        
        print(f"\n{'='*60}")
        print(f"COMPARISON RESULTS:")
        print(f"{'='*60}")
        print(f"Common neighbors: {list(common)}")
        print(f"Unique to context 1: {list(unique1)}")
        print(f"Unique to context 2: {list(unique2)}")
        
        return {
            'common': list(common),
            'unique_context1': list(unique1),
            'unique_context2': list(unique2),
            'context1_neighbors': neighbors1,
            'context2_neighbors': neighbors2
        }
    
    return None
def visualize_word_in_text_subgraph(word, text, vocab_graph, token2idx, tokenizer=tokenizer, top_n=15, highlight_word=True):
    """
    Visualize a word's neighbors within a text-specific subgraph.
    
    Args:
        word (str): Word to highlight and analyze
        text (str): Text to create subgraph from
        vocab_graph (Data): Full vocabulary graph
        token2idx (dict): Token-to-index mapping
        tokenizer: BERT tokenizer
        top_n (int): Number of top edges to show
        highlight_word (bool): Whether to highlight the target word
    """
    
    print(f"Creating visualization for word '{word}' in text context...")
    
    # Create subgraph from text
    try:
        subgraph = get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer)
        print(f"✓ Created subgraph with {subgraph.x.size(0)} nodes and {subgraph.edge_index.size(1)} edges")
    except Exception as e:
        print(f"✗ Error creating subgraph: {e}")
        return
    
    # Get tokens and create mappings
    tokens = tokenizer.tokenize(text.lower())
    present_tokens = [t for t in tokens if t in token2idx]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in present_tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    
    # Create mapping from subgraph indices to tokens
    idx_map = {i: tok for i, tok in enumerate(unique_tokens)}
    
    # Convert to NetworkX graph
    G = to_networkx(subgraph, to_undirected=True)
    
    # Check if target word is in the subgraph
    target_word_idx = None
    if word in unique_tokens:
        target_word_idx = unique_tokens.index(word)
    
    # Filter to top edges by weight
    if hasattr(subgraph, 'edge_attr') and subgraph.edge_attr is not None and len(G.edges()) > 0:
        edge_list = list(G.edges())
        edge_weights = subgraph.edge_attr.tolist()
        
        # Sort edges by absolute weight (highest first)
        edge_data = sorted(zip(edge_list, edge_weights), key=lambda x: abs(x[1]), reverse=True)
        top_edges = edge_data[:min(top_n, len(edge_data))]
        
        # Create filtered graph
        G_filtered = nx.Graph()
        nodes_in_edges = set()
        
        for (u, v), w in top_edges:
            nodes_in_edges.add(u)
            nodes_in_edges.add(v)
            G_filtered.add_edge(u, v, weight=w)
        
        # If target word is not in filtered graph, add it and its top edges
        if target_word_idx is not None and target_word_idx not in nodes_in_edges:
            # Find edges involving target word
            target_edges = []
            for (u, v), w in edge_data:
                if u == target_word_idx or v == target_word_idx:
                    target_edges.append(((u, v), w))
            
            # Add top 3 edges involving target word
            for (u, v), w in target_edges[:3]:
                G_filtered.add_edge(u, v, weight=w)
                nodes_in_edges.add(u)
                nodes_in_edges.add(v)
        
        # Update mappings for filtered graph
        filtered_idx_map = {node: idx_map[node] for node in G_filtered.nodes() if node in idx_map}
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G_filtered, seed=42, k=2, iterations=50)
        
        # Prepare node colors
        node_colors = []
        node_sizes = []
        for node in G_filtered.nodes():
            if highlight_word and node == target_word_idx:
                node_colors.append('red')  # Highlight target word
                node_sizes.append(1500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(800)
        
        # Prepare edge properties
        edge_weights = []
        edge_colors = []
        edge_widths = []
        
        for u, v in G_filtered.edges():
            weight = G_filtered[u][v]['weight']
            edge_weights.append(weight)
            edge_colors.append('green' if weight > 0 else 'red')
            edge_widths.append(max(0.5, 3 * abs(weight)))
        
        # Draw graph
        nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)
        nx.draw_networkx_labels(G_filtered, pos, labels=filtered_idx_map, font_size=11, font_weight='bold')
        
        # Add edge labels
        edge_labels = {}
        for (u, v), weight in zip(G_filtered.edges(), edge_weights):
            edge_labels[(u, v)] = f"{weight:.3f}"
        
        nx.draw_networkx_edge_labels(G_filtered, pos, edge_labels=edge_labels, font_size=9,
                                   bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        # Title and styling
        title = f"Word '{word}' in Text Subgraph"
        if target_word_idx is not None:
            title += f" (highlighted in red)"
        plt.title(title, fontsize=16, pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=4, label='Positive correlation'),
            Line2D([0], [0], color='red', lw=4, label='Negative correlation')
        ]
        if highlight_word and target_word_idx is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                        markersize=10, label=f"Target word: '{word}'"))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Show text preview
        text_preview = text[:150] + "..." if len(text) > 150 else text
        plt.figtext(0.02, 0.02, f"Text: {text_preview}", fontsize=10, style='italic', wrap=True)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print neighbor analysis
        if target_word_idx is not None:
            print(f"\nNeighbors of '{word}' in this subgraph:")
            neighbors = []
            for u, v in G_filtered.edges():
                if u == target_word_idx and v in filtered_idx_map:
                    weight = G_filtered[u][v]['weight']
                    neighbors.append((filtered_idx_map[v], weight))
                elif v == target_word_idx and u in filtered_idx_map:
                    weight = G_filtered[u][v]['weight']
                    neighbors.append((filtered_idx_map[u], weight))
            
            neighbors.sort(key=lambda x: abs(x[1]), reverse=True)
            for neighbor, weight in neighbors[:5]:
                print(f"  {neighbor}: {weight:.4f}")
        else:
            print(f"Word '{word}' not found in subgraph")
    
    else:
        print("No edge weights available for visualization")


def visualize_text_subgraph_overview(text, vocab_graph, token2idx, tokenizer=tokenizer, top_n=20):
    """
    Create an overview visualization of the entire text subgraph.
    
    Args:
        text (str): Text to create subgraph from
        vocab_graph (Data): Full vocabulary graph
        token2idx (dict): Token-to-index mapping
        tokenizer: BERT tokenizer
        top_n (int): Number of top edges to show
    """
    
    print(f"Creating overview visualization for text subgraph...")
    
    # Create subgraph
    try:
        subgraph = get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer)
        print(f"✓ Created subgraph with {subgraph.x.size(0)} nodes and {subgraph.edge_index.size(1)} edges")
    except Exception as e:
        print(f"✗ Error creating subgraph: {e}")
        return
    
    # Get tokens and create mappings
    tokens = tokenizer.tokenize(text.lower())
    present_tokens = [t for t in tokens if t in token2idx]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in present_tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    
    idx_map = {i: tok for i, tok in enumerate(unique_tokens)}
    
    # Convert to NetworkX and filter
    G = to_networkx(subgraph, to_undirected=True)
    
    if hasattr(subgraph, 'edge_attr') and subgraph.edge_attr is not None and len(G.edges()) > 0:
        edge_list = list(G.edges())
        edge_weights = subgraph.edge_attr.tolist()
        
        # Sort and filter edges
        edge_data = sorted(zip(edge_list, edge_weights), key=lambda x: abs(x[1]), reverse=True)
        top_edges = edge_data[:min(top_n, len(edge_data))]
        
        # Create filtered graph
        G_filtered = nx.Graph()
        for (u, v), w in top_edges:
            G_filtered.add_edge(u, v, weight=w)
        
        # Layout and visualization
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G_filtered, seed=42, k=2.5, iterations=50)
        
        # Node properties
        node_degrees = dict(G_filtered.degree())
        max_degree = max(node_degrees.values()) if node_degrees else 1
        
        node_sizes = [300 + 700 * (node_degrees.get(node, 0) / max_degree) for node in G_filtered.nodes()]
        node_colors = [node_degrees.get(node, 0) for node in G_filtered.nodes()]
        
        # Edge properties
        edge_weights = [G_filtered[u][v]['weight'] for u, v in G_filtered.edges()]
        edge_colors = ['green' if w > 0 else 'red' for w in edge_weights]
        edge_widths = [max(0.5, 3 * abs(w)) for w in edge_weights]
        
        # Draw graph
        nodes = nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, node_color=node_colors, 
                                     cmap=plt.cm.viridis, alpha=0.8)
        nx.draw_networkx_edges(G_filtered, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6)
        
        # Labels for high-degree nodes only
        high_degree_nodes = {node: idx_map[node] for node in G_filtered.nodes() 
                           if node in idx_map and node_degrees.get(node, 0) > max_degree * 0.3}
        
        nx.draw_networkx_labels(G_filtered, pos, labels=high_degree_nodes, font_size=10, font_weight='bold')
        
        # Title and colorbar
        plt.title("Text Subgraph Overview\n(Node size = degree, Color = degree)", fontsize=16, pad=20)
        plt.colorbar(nodes, label='Node Degree')
        
        # Text preview
        text_preview = text[:200] + "..." if len(text) > 200 else text
        plt.figtext(0.02, 0.02, f"Text: {text_preview}", fontsize=10, style='italic', wrap=True)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nSubgraph Statistics:")
        print(f"  - Total nodes: {G_filtered.number_of_nodes()}")
        print(f"  - Total edges: {G_filtered.number_of_edges()}")
        print(f"  - Average degree: {2 * G_filtered.number_of_edges() / G_filtered.number_of_nodes():.2f}")
        
        # Top degree nodes
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  - Top degree nodes: {[(idx_map.get(node, f'Node_{node}'), degree) for node, degree in top_nodes]}")
    
    else:
        print("No edge weights available for visualization")


# Example usage function
def example_visualizations():
    """Example usage of visualization functions"""
    
    vocab_graph = x_trian_graphr[0]
    token2idx = x_trian_graphr[1]
    sample_text = x_test[0][:500]
    
    print("EXAMPLE VISUALIZATIONS:")
    print("="*50)
    
    # 1. Visualize specific word in text
    print("\n1. Visualizing word 'trump' in text context:")
    visualize_word_in_text_subgraph("trump", sample_text, vocab_graph, token2idx, top_n=10)
    
    # 2. Overview of text subgraph
    print("\n2. Overview of text subgraph:")
    visualize_text_subgraph_overview(sample_text, vocab_graph, token2idx, top_n=15)


def compute_word_cooccurrence_distribution(word, texts, tokenizer, window_size=10, min_count=1):
    """
    Compute the co-occurrence probability distribution for a specific word in given texts.
    
    Args:
        word (str): Target word to analyze
        texts (list): List of text documents
        tokenizer: BERT tokenizer
        window_size (int): Window size for co-occurrence
        min_count (int): Minimum count threshold
    
    Returns:
        dict: Probability distribution P(j|word) where j are co-occurring words
    """
    # Tokenize all texts
    tokenized_docs = tokenize_bert(texts, tokenizer)
    
    # Count co-occurrences with the target word
    word_cooccurrences = defaultdict(int)
    total_cooccurrences = 0
    
    for tokens in tokenized_docs:
        for i, token in enumerate(tokens):
            if token == word:
                # Get window around the target word
                start = max(0, i - window_size // 2)
                end = min(len(tokens), i + window_size // 2 + 1)
                window_tokens = tokens[start:end]
                
                # Count co-occurrences (excluding the target word itself)
                for cooccur_token in window_tokens:
                    if cooccur_token != word:
                        word_cooccurrences[cooccur_token] += 1
                        total_cooccurrences += 1
    
    # Convert to probability distribution
    if total_cooccurrences == 0:
        return {}
    
    prob_distribution = {}
    for cooccur_word, count in word_cooccurrences.items():
        if count >= min_count:
            prob_distribution[cooccur_word] = count / total_cooccurrences
    
    return prob_distribution

def compute_kl_divergence(P, Q, smoothing=1e-10):
    """
    Compute KL divergence D_KL(P || Q) between two probability distributions.
    
    Args:
        P (dict): First probability distribution
        Q (dict): Second probability distribution  
        smoothing (float): Smoothing parameter to avoid log(0)
    
    Returns:
        float: KL divergence value
    """
    if not P or not Q:
        return float('inf')
    
    # Get all unique words from both distributions
    all_words = set(P.keys()) | set(Q.keys())
    
    if not all_words:
        return 0.0
    
    kl_div = 0.0
    for word in all_words:
        p_prob = P.get(word, 0) + smoothing
        q_prob = Q.get(word, 0) + smoothing
        
        kl_div += p_prob * np.log(p_prob / q_prob)
    
    return kl_div

def test_context_specific_divergence(words_to_test, real_texts, fake_texts, tokenizer, 
                                   window_size=10, min_count=2, threshold_epsilon=0.1):
    """
    Test the Context-Specific Association Divergence theorem for multiple words.
    
    Args:
        words_to_test (list): List of words to analyze
        real_texts (list): Real news texts
        fake_texts (list): Fake news texts
        tokenizer: BERT tokenizer
        window_size (int): Window size for co-occurrence
        min_count (int): Minimum count threshold
        threshold_epsilon (float): Divergence threshold ε
    
    Returns:
        pd.DataFrame: Results with divergence values and significance tests
    """
    results = []
    
    print(f"Testing Context-Specific Association Divergence for {len(words_to_test)} words...")
    print("="*70)
    
    for i, word in enumerate(words_to_test):
        print(f"Processing word {i+1}/{len(words_to_test)}: '{word}'")
        
        try:
            # Compute distributions for real and fake contexts
            P_real = compute_word_cooccurrence_distribution(word, real_texts, tokenizer, window_size, min_count)
            P_fake = compute_word_cooccurrence_distribution(word, fake_texts, tokenizer, window_size, min_count)
            
            if not P_real or not P_fake:
                print(f"  ⚠️  Insufficient data for '{word}', skipping...")
                continue
            
            # Compute KL divergences in both directions
            kl_real_fake = compute_kl_divergence(P_real, P_fake)
            kl_fake_real = compute_kl_divergence(P_fake, P_real)
            
            # Compute symmetric divergence (average)
            symmetric_kl = (kl_real_fake + kl_fake_real) / 2
            
            # Test against threshold
            exceeds_threshold = kl_real_fake > threshold_epsilon
            
            # Compute additional statistics
            real_vocab_size = len(P_real)
            fake_vocab_size = len(P_fake)
            common_vocab = len(set(P_real.keys()) & set(P_fake.keys()))
            
            # Store results
            result = {
                'word': word,
                'kl_real_to_fake': kl_real_fake,
                'kl_fake_to_real': kl_fake_real,
                'symmetric_kl': symmetric_kl,
                'exceeds_threshold': exceeds_threshold,
                'real_vocab_size': real_vocab_size,
                'fake_vocab_size': fake_vocab_size,
                'common_vocab_size': common_vocab,
                'P_real': P_real,
                'P_fake': P_fake
            }
            
            results.append(result)
            
            # Print progress
            status = "✅ SIGNIFICANT" if exceeds_threshold else "❌ Not significant"
            print(f"  KL(Real||Fake): {kl_real_fake:.4f} | Threshold: {threshold_epsilon} | {status}")
            
        except Exception as e:
            print(f"  ❌ Error processing '{word}': {e}")
            continue
    
    # Convert to DataFrame
    if results:
        df_results = pd.DataFrame(results)
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print(f"SUMMARY RESULTS:")
        print(f"{'='*70}")
        print(f"Total words tested: {len(results)}")
        print(f"Words with significant divergence (> {threshold_epsilon}): {df_results['exceeds_threshold'].sum()}")
        print(f"Percentage significant: {df_results['exceeds_threshold'].mean()*100:.1f}%")
        print(f"Mean KL divergence: {df_results['kl_real_to_fake'].mean():.4f}")
        print(f"Max KL divergence: {df_results['kl_real_to_fake'].max():.4f}")
        print(f"Min KL divergence: {df_results['kl_real_to_fake'].min():.4f}")
        
        return df_results
    else:
        print("No results obtained!")
        return pd.DataFrame()

def permutation_test_divergence(word, real_texts, fake_texts, tokenizer, n_permutations=1000, 
                               window_size=10, min_count=2):
    """
    Perform permutation test to determine if divergence is statistically significant.
    
    Args:
        word (str): Word to test
        real_texts (list): Real news texts
        fake_texts (list): Fake news texts
        tokenizer: BERT tokenizer
        n_permutations (int): Number of permutation iterations
        window_size (int): Window size for co-occurrence
        min_count (int): Minimum count threshold
    
    Returns:
        dict: Test results including p-value and significance
    """
    print(f"Running permutation test for word '{word}' with {n_permutations} permutations...")
    
    # Compute original divergence
    P_real = compute_word_cooccurrence_distribution(word, real_texts, tokenizer, window_size, min_count)
    P_fake = compute_word_cooccurrence_distribution(word, fake_texts, tokenizer, window_size, min_count)
    
    if not P_real or not P_fake:
        return {'error': 'Insufficient data for permutation test'}
    
    original_kl = compute_kl_divergence(P_real, P_fake)
    
    # Combine all texts and shuffle labels
    all_texts = real_texts + fake_texts
    original_labels = [0] * len(real_texts) + [1] * len(fake_texts)  # 0=real, 1=fake
    
    permuted_kls = []
    
    for i in range(n_permutations):
        if i % 100 == 0:
            print(f"  Permutation {i}/{n_permutations}")
        
        # Shuffle labels
        shuffled_labels = np.random.permutation(original_labels)
        
        # Split based on shuffled labels
        shuffled_real = [all_texts[j] for j in range(len(all_texts)) if shuffled_labels[j] == 0]
        shuffled_fake = [all_texts[j] for j in range(len(all_texts)) if shuffled_labels[j] == 1]
        
        # Compute distributions for shuffled data
        P_shuffled_real = compute_word_cooccurrence_distribution(word, shuffled_real, tokenizer, window_size, min_count)
        P_shuffled_fake = compute_word_cooccurrence_distribution(word, shuffled_fake, tokenizer, window_size, min_count)
        
        if P_shuffled_real and P_shuffled_fake:
            shuffled_kl = compute_kl_divergence(P_shuffled_real, P_shuffled_fake)
            permuted_kls.append(shuffled_kl)
    
    if not permuted_kls:
        return {'error': 'No valid permutations generated'}
    
    # Calculate p-value
    p_value = np.mean(np.array(permuted_kls) >= original_kl)
    
    results = {
        'word': word,
        'original_kl': original_kl,
        'permuted_kls': permuted_kls,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'n_permutations': len(permuted_kls)
    }
    
    print(f"  Original KL: {original_kl:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    return results

def visualize_divergence_results(df_results, save_path=None):
    """
    Create comprehensive visualizations of divergence test results.
    
    Args:
        df_results (pd.DataFrame): Results from test_context_specific_divergence
        save_path (str): Optional path to save plots
    """
    if df_results.empty:
        print("No results to visualize!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Context-Specific Association Divergence Analysis', fontsize=16, fontweight='bold')
    
    # 1. KL Divergence Distribution
    ax1 = axes[0, 0]
    ax1.hist(df_results['kl_real_to_fake'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df_results['kl_real_to_fake'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_results["kl_real_to_fake"].mean():.3f}')
    ax1.set_xlabel('KL Divergence D_KL(P_real || P_fake)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of KL Divergences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Threshold Comparison
    ax2 = axes[0, 1]
    threshold_counts = df_results['exceeds_threshold'].value_counts()
    colors = ['lightcoral', 'lightgreen']
    labels = ['Below Threshold', 'Above Threshold']
    ax2.pie(threshold_counts.values, labels=[f'{labels[i]}\n({threshold_counts.values[i]} words)' 
                                           for i in range(len(threshold_counts))], 
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Words Exceeding Divergence Threshold')
    
    # 3. Top Words by Divergence
    ax3 = axes[1, 0]
    top_words = df_results.nlargest(10, 'kl_real_to_fake')
    bars = ax3.barh(range(len(top_words)), top_words['kl_real_to_fake'], color='lightblue')
    ax3.set_yticks(range(len(top_words)))
    ax3.set_yticklabels(top_words['word'])
    ax3.set_xlabel('KL Divergence')
    ax3.set_title('Top 10 Words by KL Divergence')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # 4. Vocabulary Size Correlation
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_results['common_vocab_size'], df_results['kl_real_to_fake'], 
                         alpha=0.6, c=df_results['kl_real_to_fake'], cmap='viridis')
    ax4.set_xlabel('Common Vocabulary Size')
    ax4.set_ylabel('KL Divergence')
    ax4.set_title('KL Divergence vs Common Vocabulary Size')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='KL Divergence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\nDETAILED STATISTICS:")
    print(f"{'='*50}")
    print(f"Total words analyzed: {len(df_results)}")
    print(f"Mean KL divergence: {df_results['kl_real_to_fake'].mean():.4f}")
    print(f"Std KL divergence: {df_results['kl_real_to_fake'].std():.4f}")
    print(f"Median KL divergence: {df_results['kl_real_to_fake'].median():.4f}")
    print(f"Min KL divergence: {df_results['kl_real_to_fake'].min():.4f}")
    print(f"Max KL divergence: {df_results['kl_real_to_fake'].max():.4f}")
    
    # Print top words
    print(f"\nTOP 5 WORDS BY DIVERGENCE:")
    print(f"{'-'*50}")
    top_5 = df_results.nlargest(5, 'kl_real_to_fake')
    for _, row in top_5.iterrows():
        print(f"{row['word']:<15} KL: {row['kl_real_to_fake']:.4f}")

def analyze_specific_word_divergence(word, real_texts, fake_texts, tokenizer, 
                                   window_size=10, min_count=2, show_top_neighbors=10):
    """
    Detailed analysis of divergence for a specific word.
    
    Args:
        word (str): Word to analyze
        real_texts (list): Real news texts  
        fake_texts (list): Fake news texts
        tokenizer: BERT tokenizer
        window_size (int): Window size for co-occurrence
        min_count (int): Minimum count threshold
        show_top_neighbors (int): Number of top neighbors to display
    
    Returns:
        dict: Detailed analysis results
    """
    print(f"DETAILED DIVERGENCE ANALYSIS FOR: '{word}'")
    print(f"{'='*60}")
    
    # Compute distributions
    P_real = compute_word_cooccurrence_distribution(word, real_texts, tokenizer, window_size, min_count)
    P_fake = compute_word_cooccurrence_distribution(word, fake_texts, tokenizer, window_size, min_count)
    
    if not P_real or not P_fake:
        print(f"Insufficient data for word '{word}'")
        return {}
    
    # Compute divergences
    kl_real_fake = compute_kl_divergence(P_real, P_fake)
    kl_fake_real = compute_kl_divergence(P_fake, P_real)
    
    # Analyze distributions
    all_neighbors = set(P_real.keys()) | set(P_fake.keys())
    
    # Find words that differ most between contexts
    difference_analysis = []
    for neighbor in all_neighbors:
        prob_real = P_real.get(neighbor, 0)
        prob_fake = P_fake.get(neighbor, 0)
        diff = abs(prob_real - prob_fake)
        ratio = (prob_real + 1e-10) / (prob_fake + 1e-10)
        
        difference_analysis.append({
            'neighbor': neighbor,
            'prob_real': prob_real,
            'prob_fake': prob_fake,
            'abs_difference': diff,
            'ratio_real_to_fake': ratio
        })
    
    # Sort by absolute difference
    difference_analysis.sort(key=lambda x: x['abs_difference'], reverse=True)
    
    # Print results
    print(f"KL Divergence D_KL(Real || Fake): {kl_real_fake:.4f}")
    print(f"KL Divergence D_KL(Fake || Real): {kl_fake_real:.4f}")
    print(f"Symmetric KL Divergence: {(kl_real_fake + kl_fake_real)/2:.4f}")
    print(f"\nVocabulary Statistics:")
    print(f"  Real context neighbors: {len(P_real)}")
    print(f"  Fake context neighbors: {len(P_fake)}")
    print(f"  Common neighbors: {len(set(P_real.keys()) & set(P_fake.keys()))}")
    
    print(f"\nTOP {show_top_neighbors} MOST DIFFERENT NEIGHBORS:")
    print(f"{'-'*70}")
    print(f"{'Neighbor':<15} {'Real Prob':<12} {'Fake Prob':<12} {'Abs Diff':<10} {'Ratio':<8}")
    print(f"{'-'*70}")
    
    for item in difference_analysis[:show_top_neighbors]:
        print(f"{item['neighbor']:<15} {item['prob_real']:<12.4f} {item['prob_fake']:<12.4f} "
              f"{item['abs_difference']:<10.4f} {item['ratio_real_to_fake']:<8.2f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top neighbors in each context
    top_real = sorted(P_real.items(), key=lambda x: x[1], reverse=True)[:10]
    top_fake = sorted(P_fake.items(), key=lambda x: x[1], reverse=True)[:10]
    
    ax1.barh(range(len(top_real)), [prob for _, prob in top_real], color='lightblue')
    ax1.set_yticks(range(len(top_real)))
    ax1.set_yticklabels([word for word, _ in top_real])
    ax1.set_xlabel('Probability')
    ax1.set_title(f'Top Neighbors of "{word}" in REAL News')
    ax1.grid(True, alpha=0.3)
    
    ax2.barh(range(len(top_fake)), [prob for _, prob in top_fake], color='lightcoral')
    ax2.set_yticks(range(len(top_fake)))
    ax2.set_yticklabels([word for word, _ in top_fake])
    ax2.set_xlabel('Probability')
    ax2.set_title(f'Top Neighbors of "{word}" in FAKE News')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'word': word,
        'kl_real_fake': kl_real_fake,
        'kl_fake_real': kl_fake_real,
        'P_real': P_real,
        'P_fake': P_fake,
        'difference_analysis': difference_analysis
    }

def run_complete_divergence_analysis(real_texts, fake_texts, tokenizer, 
                                   target_words=None, n_random_words=20, 
                                   threshold_epsilon=0.1, run_permutation_tests=False):
    """
    Run complete analysis to prove Context-Specific Association Divergence theorem.
    
    Args:
        real_texts (list): Real news texts
        fake_texts (list): Fake news texts  
        tokenizer: BERT tokenizer
        target_words (list): Specific words to test (optional)
        n_random_words (int): Number of random words to test if target_words not provided
        threshold_epsilon (float): Divergence threshold ε
        run_permutation_tests (bool): Whether to run statistical significance tests
    
    Returns:
        dict: Complete analysis results
    """
    print("RUNNING COMPLETE CONTEXT-SPECIFIC ASSOCIATION DIVERGENCE ANALYSIS")
    print("="*80)
    
    # Determine words to test
    if target_words is None:
        # Get common words from both corpora
        print("Extracting common vocabulary...")
        real_vocab = set()
        fake_vocab = set()
        
        for text in real_texts[:100]:  # Sample for efficiency
            tokens = tokenizer.tokenize(text.lower())
            real_vocab.update(tokens)
        
        for text in fake_texts[:100]:  # Sample for efficiency  
            tokens = tokenizer.tokenize(text.lower())
            fake_vocab.update(tokens)
        
        common_vocab = real_vocab & fake_vocab
        # Filter out very common and very rare words
        target_words = [w for w in common_vocab if len(w) > 2 and w.isalpha()]
        target_words = list(np.random.choice(target_words, min(n_random_words, len(target_words)), replace=False))
        
        print(f"Selected {len(target_words)} random words from common vocabulary")
    
    print(f"Testing words: {target_words}")
    
    # Run main divergence test
    df_results = test_context_specific_divergence(
        target_words, real_texts, fake_texts, tokenizer, 
        threshold_epsilon=threshold_epsilon
    )
    
    if df_results.empty:
        print("No results obtained from divergence test!")
        return {}
    
    # Visualize results
    visualize_divergence_results(df_results)
    
    results = {
        'main_results': df_results,
        'target_words': target_words,
        'threshold_epsilon': threshold_epsilon
    }
    
    # Run permutation tests for top divergent words (if requested)
    if run_permutation_tests:
        print(f"\nRunning permutation tests for top 3 words...")
        top_words = df_results.nlargest(3, 'kl_real_to_fake')['word'].tolist()
        
        permutation_results = []
        for word in top_words:
            perm_result = permutation_test_divergence(word, real_texts, fake_texts, tokenizer)
            if 'error' not in perm_result:
                permutation_results.append(perm_result)
        
        results['permutation_tests'] = permutation_results
    
    # Detailed analysis of most divergent word
    if len(df_results) > 0:
        most_divergent_word = df_results.loc[df_results['kl_real_to_fake'].idxmax(), 'word']
        print(f"\nDetailed analysis of most divergent word: '{most_divergent_word}'")
        detailed_analysis = analyze_specific_word_divergence(
            most_divergent_word, real_texts, fake_texts, tokenizer
        )
        results['detailed_analysis'] = detailed_analysis
    
    # Final summary
    significant_count = df_results['exceeds_threshold'].sum()
    total_count = len(df_results)
    
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - PROOF OF CONTEXT-SPECIFIC ASSOCIATION DIVERGENCE")
    print(f"{'='*80}")
    print(f"Theorem: D_KL(P_{{i|real}} || P_{{i|fake}}) > ε")
    print(f"Threshold ε = {threshold_epsilon}")
    print(f"Words tested: {total_count}")
    print(f"Words with significant divergence: {significant_count}")
    print(f"Percentage proving theorem: {significant_count/total_count*100:.1f}%")
    
    if significant_count > 0:
        print(f"✅ THEOREM SUPPORTED: {significant_count}/{total_count} words show significant context-specific divergence")
    else:
        print(f"❌ THEOREM NOT SUPPORTED: No words show significant divergence above threshold")
    
    print(f"Mean divergence: {df_results['kl_real_to_fake'].mean():.4f}")
    print(f"Max divergence: {df_results['kl_real_to_fake'].max():.4f}")
    
    return results

# Example usage function
def example_divergence_analysis():
    """
    Example of how to run the divergence analysis with your existing data.
    """
    # Use your existing data
    global news_o, news_1, tokenizer
    
    # Define some target words to test
    target_words = ['president', 'officials', 'reported', 'confirmed', 'sources', 
                   'investigation', 'statement', 'trump', 'administration', 'according']
    
    print("Example: Running divergence analysis...")
    
    # Sample data for faster testing
    real_sample = news_o[:200] if len(news_o) > 200 else news_o
    fake_sample = news_1[:200] if len(news_1) > 200 else news_1
    
    # Run complete analysis
    results = run_complete_divergence_analysis(
        real_sample, fake_sample, tokenizer,
        target_words=target_words,
        threshold_epsilon=0.1,
        run_permutation_tests=True  # Set to False for faster execution
    )
    
    return results

# Uncomment to run the example:
# results = example_divergence_analysis()
# Uncomment to run examples:
# example_visualizations()
obj = 'gossipcop'
# idx2graph = {}
# fout = open('test_res'+obj+'.Rograph', 'wb')
# for i,ele in enumerate(x_test_res):
#     #print(i)
#     adj_matrix,_ = construct_text_graph(ele[:600])
#     idx2graph[i] = adj_matrix
# pickle.dump(idx2graph, fout)        
# fout.close()
train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))

#restyle_dict = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_A.pkl', 'rb'))
# alternatively, switch to loading other adversarial test sets with '_test_adv_[B/C/D].pkl'

x_train, y_train = train_dict['news'], train_dict['labels']
x_test, y_test = test_dict['news'], test_dict['labels']

#x_test_res = restyle_dict['news']

news_o,news_1=split_news_by_binary_label(train_dict)

x_trian_graphr=load_vocab_graphs('gr/vocab_r_'+obj+'.Rograph',news_o)
x_trian_graphf=load_vocab_graphs('gr/vocab_f_'+obj+'.Rograph',news_1)
# vocab_graph = x_trian_graphf[0]  # Real news vocabulary graph
# token2idx = x_trian_graphf[1]
# print_word_neighbors_in_text_detailed("obama", x_test[0][:500], x_trian_graphf[0], x_trian_graphf[1])
# word = "president"
# neighbors = print_word_neighbors_detailed(word, vocab_graph, token2idx, top_k=15, min_weight=0.01)
# print(x_test[0][:350])
# visualize_vocab_subgraph(x_test[0][:250], x_trian_graphf[0], x_trian_graphf[1], tokenizer)
# with open('train_label_article.txt', 'w', encoding='utf-8') as f:
#     for label, article in zip(y_train, x_train):
#         # Clean the article text (remove newlines and tabs)
#         article_clean = article.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
#         f.write(f"{label}\t{article_clean}\n")

# print(f"Saved {len(x_train)} training samples (label + article) to 'train_label_article.txt'")

# # Save test data: label + article (tab-separated)
# with open('test_label_article.txt', 'w', encoding='utf-8') as f:
#     for label, article in zip(y_test, x_test):
#         # Clean the article text (remove newlines and tabs)
#         article_clean = article.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
#         f.write(f"{label}\t{article_clean}\n")

# print(f"Saved {len(x_test)} test samples (label + article) to 'test_label_article.txt'")

# # Alternative: Save with pipe separator (if labels contain spaces)
# with open('train_label_article_pipe.txt', 'w', encoding='utf-8') as f:
#     for label, article in zip(y_train, x_train):
#         article_clean = article.replace('\n', ' ').replace('\r', ' ')
#         f.write(f"{label}|{article_clean}\n")



# print(f"Saved {len(x_train)} training samples with pipe separator to 'train_label_article_pipe.txt'")
def run_divergence_proof_analysis():
    """
    Main function to run the empirical proof of Context-Specific Association Divergence
    """
    print("EMPIRICAL PROOF OF CONTEXT-SPECIFIC ASSOCIATION DIVERGENCE THEOREM")
    print("="*80)
    
    # Use your existing real and fake news data
    global news_o, news_1, tokenizer
    
    # Sample data for computational efficiency (adjust size as needed)
    real_texts = news_o[:300] if len(news_o) > 300 else news_o
    fake_texts = news_1[:300] if len(news_1) > 300 else news_1
    
    print(f"Using {len(real_texts)} real news articles and {len(fake_texts)} fake news articles")
    
    # Define politically relevant words to test

   # brad pitt made rare late night appearance last night \( may 16 \) episode late show stephen colbert , participated another hilarious installment big questions even bigger stars ! making joke , 53 year old war machine actor laid back blanket stephen , 53 , asked life deep , philosophical questions photos check latest pics brad pitt think mathematics underlying structure universe , invent \? brad pondered stephen replied , sure numbers underlying reality , one thing absolutely true solid eight brad also poked fun biggest films stephen tried make laugh silly nicknames like colby cheese , stevie c brad pitt en watch full segment brad pitt big questions even bigger stars
    political_words = [
        'pitt', 'appearance', 'officials', 'episode',
        'making', 'joke', 'laid', 'back', 'asked', 'life',
    ]
    
    # Also test some general news words
    general_words = [
        'made', 'told', 'late', 'news', 'night', 'story', 'media',
        'press', 'even','bigger'  'stars', 'update'

    ]
    
    # Combine word lists
    test_words = political_words + general_words
    
    print(f"Testing {len(test_words)} words for context-specific divergence...")
    
    # Run the complete analysis
    results = run_complete_divergence_analysis(
        real_texts=real_texts,
        fake_texts=fake_texts,
        tokenizer=tokenizer,
        target_words=test_words,
        threshold_epsilon=0.05,  # Lower threshold for more sensitive detection
        run_permutation_tests=True  # Set to False for faster execution
    )
    
    return results

def analyze_top_divergent_words(results, top_n=5):
    """
    Detailed analysis of the top N most divergent words
    """
    if 'main_results' not in results or results['main_results'].empty:
        print("No results to analyze!")
        return
    
    df_results = results['main_results']
    top_words = df_results.nlargest(top_n, 'kl_real_to_fake')
    
    print(f"\nDETAILED ANALYSIS OF TOP {top_n} DIVERGENT WORDS")
    print("="*60)
    
    for _, row in top_words.iterrows():
        word = row['word']
        kl_div = row['kl_real_to_fake']
        
        print(f"\nAnalyzing word: '{word}' (KL divergence: {kl_div:.4f})")
        print("-" * 50)
        
        # Detailed analysis
        detailed = analyze_specific_word_divergence(
            word, news_o[:200], news_1[:200], tokenizer, show_top_neighbors=8
        )

def compare_word_contexts_example():
    """
    Example of comparing specific words between real and fake contexts
    """
    print("\nEXAMPLE: Comparing word contexts...")
    
    # Test a few specific high-impact words
    test_words =  [
        'made', 'told', 'late', 'news', 'night', 'story', 'media',
        'press', 'even','bigger'  'stars', 'update'

    ]
    
    for word in test_words:
        print(f"\n" + "="*60)
        print(f"CONTEXT COMPARISON FOR: '{word}'")
        print("="*60)
        
        # Get distributions
        P_real = compute_word_cooccurrence_distribution(word, news_o[:100], tokenizer)
        P_fake = compute_word_cooccurrence_distribution(word, news_1[:100], tokenizer)
        
        if P_real and P_fake:
            kl_div = compute_kl_divergence(P_real, P_fake)
            
            print(f"KL Divergence: {kl_div:.4f}")
            
            # Top neighbors in each context
            top_real = sorted(P_real.items(), key=lambda x: x[1], reverse=True)[:5]
            top_fake = sorted(P_fake.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\nTop neighbors in REAL news:")
            for neighbor, prob in top_real:
                print(f"  {neighbor}: {prob:.4f}")
            
            print(f"\nTop neighbors in FAKE news:")
            for neighbor, prob in top_fake:
                print(f"  {neighbor}: {prob:.4f}")
        else:
            print(f"Insufficient data for word '{word}'")

def save_divergence_results(results, filename='divergence_analysis_results.pkl'):
    """
    Save analysis results for later use
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    # Run the main analysis
    print("Starting Context-Specific Association Divergence Analysis...")
    
    # Make sure data is loaded
    if 'news_o' not in globals() or 'news_1' not in globals():
        print("Loading data...")
        obj = 'gossipcop'
        train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
        news_o, news_1 = split_news_by_binary_label(train_dict)
    
    # Run the main proof analysis
    results = run_divergence_proof_analysis()
    
    # Analyze top divergent words in detail
    analyze_top_divergent_words(results, top_n=3)
    
    # Compare specific word contexts
    compare_word_contexts_example()
    
    # Save results
    save_divergence_results(results)
    
    print("\n" + "="*80)
    print("DIVERGENCE ANALYSIS COMPLETE!")
    print("="*80)
