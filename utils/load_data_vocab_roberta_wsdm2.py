import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import spacy
from transformers import RobertaTokenizer, RobertaModel
import os
import math
from collections import defaultdict, Counter
from itertools import combinations
import networkx as nx
from torch_geometric.utils import to_networkx, subgraph

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def tokenize_roberta(texts, tokenizer):
    """Tokenize texts using RoBERTa tokenizer"""
    tokenized_docs = []
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        tokenized_docs.append(tokens)
    return tokenized_docs


def build_vocab_and_cooc(tokenized_docs, window_size=10, min_count=0):
    """Build vocabulary and co-occurrence statistics with configurable window size"""
    vocab_counter = Counter()
    co_occurrence = defaultdict(int)
    total_windows = 0

    print(f"Building vocabulary with window_size={window_size}, min_count={min_count}")
    
    for doc_idx, tokens in enumerate(tokenized_docs):
        if doc_idx % 1000 == 0:
            print(f"Processing document {doc_idx}/{len(tokenized_docs)}")
            
        for i in range(len(tokens)):
            # Use the configurable window size
            window = tokens[i:i+window_size]
            unique = set(window)
            total_windows += 1

            for token in unique:
                vocab_counter[token] += 1
            for t1, t2 in combinations(sorted(unique), 2):
                co_occurrence[(t1, t2)] += 1

    vocab = {w for w, c in vocab_counter.items() if c >= min_count}
    token2idx = {token: idx for idx, token in enumerate(sorted(vocab))}
    
    print(f"Built vocabulary: {len(vocab)} tokens, {len(co_occurrence)} co-occurrences")
    print(f"Window size effect: larger windows capture broader context, smaller windows focus on local relationships")

    return vocab, token2idx, vocab_counter, co_occurrence, total_windows


def npmi(w1, w2, vocab_counter, co_occurrence, total_windows):
    """Calculate Normalized Pointwise Mutual Information"""
    p_x = vocab_counter[w1] / total_windows
    p_y = vocab_counter[w2] / total_windows
    p_xy = co_occurrence[(w1, w2)] / total_windows
    if p_xy == 0:
        return None
    pmi = math.log(p_xy / (p_x * p_y) + 1e-10)
    return pmi / (-math.log(p_xy + 1e-10))


def compute_npmi_edges(token2idx, vocab_counter, co_occurrence, total_windows):
    """Compute edges based on NPMI scores"""
    edges = []
    weights = []

    for (w1, w2), _ in co_occurrence.items():
        if w1 in token2idx and w2 in token2idx:
            score = npmi(w1, w2, vocab_counter, co_occurrence, total_windows)
            if score and score > 0:
                edges.append((token2idx[w1], token2idx[w2]))
                weights.append(score)

    return edges, weights


def filter_edges_by_threshold(graph, threshold):
    """
    Filter edges based on edge attribute threshold.
    Removes edges where edge_attr < threshold.
    
    Args:
        graph: PyTorch Geometric Data object
        threshold: float, minimum edge weight to keep
    
    Returns:
        Data object with filtered edges
    """
    if graph.edge_attr is None or graph.edge_index is None:
        print("Warning: Graph has no edges or edge attributes")
        return graph
    
    # Find edges with weights >= threshold
    edge_mask = graph.edge_attr >= threshold
    
    # Filter edges and edge attributes
    filtered_edge_index = graph.edge_index[:, edge_mask]
    filtered_edge_attr = graph.edge_attr[edge_mask]
    
    #print(f"Edge filtering: {graph.edge_index.size(1)} -> {filtered_edge_index.size(1)} edges (threshold={threshold})")
    
    # Return new graph with filtered edges
    return Data(
        x=graph.x,
        edge_index=filtered_edge_index,
        edge_attr=filtered_edge_attr
    )


def build_pyg_graph_efficient(texts, tokenizer=tokenizer, window_size=10, min_count=1, 
                            node_feature_dim=768, use_roberta_init=False):
    """Efficient graph building with proper node feature dimensions"""
    tokenized = tokenize_roberta(texts, tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(tokenized, window_size, min_count)
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    
    num_tokens = len(token2idx)
    print(f"Building efficient graph: {num_tokens} tokens, {len(edges)} edges")
    print(f"Node feature dimension: {node_feature_dim}")
    
    if not edges:
        print("Warning: No edges found, creating graph with isolated nodes")
        x = torch.randn(max(1, num_tokens), node_feature_dim) * 0.1
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

    # Create edge tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)

    # Create appropriately sized node features
    if use_roberta_init:
        print("Initializing with RoBERTa embeddings (this may take a while)...")
        x = initialize_with_roberta_embeddings(token2idx, tokenizer, node_feature_dim)
    else:
        # Simple random initialization (much faster)
        x = torch.randn(num_tokens, node_feature_dim) * 0.1
        print(f"Initialized with random features: {x.shape}")

    # Verify everything is consistent
    assert x.size(0) == num_tokens, f"Node features size {x.size(0)} != vocab size {num_tokens}"
    assert edge_index.max().item() < num_tokens, f"Edge index {edge_index.max().item()} >= vocab size {num_tokens}"
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx


def build_pyg_graph_efficient_with_filtering(texts, tokenizer=tokenizer, window_size=10, min_count=1, 
                                           node_feature_dim=768, use_roberta_init=False, edge_threshold=0.0):
    """Efficient graph building with configurable window size and edge filtering"""
    print(f"Graph building parameters:")
    print(f"  - Window size: {window_size}")
    print(f"  - Edge threshold: {edge_threshold}")
    print(f"  - Min count: {min_count}")
    print(f"  - Node feature dim: {node_feature_dim}")
    
    tokenized = tokenize_roberta(texts, tokenizer)
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc(
        tokenized, window_size=window_size, min_count=min_count
    )
    edges, weights = compute_npmi_edges(token2idx, counter, cooc, total_windows)
    
    num_tokens = len(token2idx)
    original_edges = len(edges)
    print(f"Initial graph: {num_tokens} tokens, {original_edges} edges")
    
    if not edges:
        print("Warning: No edges found, creating graph with isolated nodes")
        x = torch.randn(max(1, num_tokens), node_feature_dim) * 0.1
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx

    # Create edge tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    # Apply edge filtering if threshold > 0
    if edge_threshold > 0.0:
        edge_mask = edge_weight >= edge_threshold
        edge_index = edge_index[:, edge_mask]
        edge_weight = edge_weight[edge_mask]
        filtered_edges = edge_weight.size(0)
        #print(f"Edge filtering: {original_edges} -> {filtered_edges} edges (kept {filtered_edges/original_edges*100:.1f}%)")
        
        if filtered_edges == 0:
            print("Warning: All edges filtered out! Consider lowering edge_threshold")
    else:
        print("No edge filtering applied")

    # Create node features
    if use_roberta_init:
        print("Initializing with RoBERTa embeddings...")
        x = initialize_with_roberta_embeddings(token2idx, tokenizer, node_feature_dim)
    else:
        x = torch.randn(num_tokens, node_feature_dim) * 0.1

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight), token2idx


def initialize_with_roberta_embeddings(token2idx, tokenizer, embedding_dim=768):
    """Initialize node features with RoBERTa embeddings"""
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    roberta_model.eval()
    
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
                    outputs = roberta_model(token_ids)
                    
                    # Use average of all token embeddings
                    token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = token_embedding[:embedding_dim]  # Ensure correct size
                    
                except Exception as e:
                    print(f"Warning: Could not get RoBERTa embedding for token '{token}': {e}")
                    # Use random initialization for problematic tokens
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = torch.randn(embedding_dim) * 0.1
    
    return x


def get_vocab_subgraph_from_text_robust(text, vocab_graph, token2idx, tokenizer):
    """Robust version of get_vocab_subgraph_from_text with better error handling"""
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
            x=torch.zeros(1, 768).to(device),
            edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
            edge_attr=torch.zeros(0).to(device)
        )


def get_vocab_subgraph_from_text_robust_with_filtering(text, vocab_graph, token2idx, tokenizer, edge_threshold=0.0):
    """Robust subgraph extraction with optional edge filtering"""
    try:
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        if not token2idx:
            raise ValueError("Empty token2idx vocabulary")
        
        # Tokenize the input text
        tokens = tokenizer.tokenize(text.lower())
        
        if not tokens:
            raise ValueError("No tokens found in text")
        
        # Map to graph node indices
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
        edge_index, edge_attr = subgraph(
            node_mask, 
            vocab_graph.edge_index, 
            edge_attr=vocab_graph.edge_attr, 
            relabel_nodes=True
        )

        # Get node features
        x = vocab_graph.x[node_mask]
        
        # Create subgraph
        subgraph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Apply edge filtering if threshold > 0
        if edge_threshold > 0.0 and edge_attr is not None:
            subgraph_data = filter_edges_by_threshold(subgraph_data, edge_threshold)

        return subgraph_data
        
    except Exception as e:
        print(f"Warning: Error in subgraph extraction: {e}")
        # Return minimal valid subgraph
        device = vocab_graph.x.device if hasattr(vocab_graph, 'x') and vocab_graph.x is not None else torch.device('cpu')
        return Data(
            x=torch.zeros(1, 768).to(device),
            edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
            edge_attr=torch.zeros(0).to(device)
        )


def split_news_by_binary_label(train_dict):
    """Split news texts into two lists based on binary labels (0 or 1)"""
    news = train_dict['news']
    labels = train_dict['labels']

    if len(news) != len(labels):
        raise ValueError("The number of news items must match the number of labels.")

    news_0 = [n for n, l in zip(news, labels) if l == 0]
    news_1 = [n for n, l in zip(news, labels) if l == 1]

    return news_0, news_1


def load_vocab_graphs(file_path, texts, force_rebuild=False):
    """Load or build vocabulary graphs with correct RoBERTa-sized features"""
    if file_exists(file_path) and not force_rebuild:
        try:
            print(f"Loading existing vocabulary graph from {file_path}")
            vocab_data = pickle.load(open(file_path, 'rb'))
            
            # Validate the data structure
            if isinstance(vocab_data, dict) and 0 in vocab_data and 1 in vocab_data:
                graph, token2idx = vocab_data[0], vocab_data[1]
                
                # Check node feature dimensions
                if graph.x is not None:
                    node_feature_dim = graph.x.size(1)
                    num_nodes = graph.x.size(0)
                    vocab_size = len(token2idx)
                    
                    print(f"Loaded graph: {num_nodes} nodes, {vocab_size} vocab size, {node_feature_dim}D features")
                    
                    # Check if we have the old problematic format
                    if node_feature_dim == vocab_size:
                        print("⚠️ Detected old identity matrix format! Forcing rebuild...")
                        force_rebuild = True
                    elif node_feature_dim != 768:
                        print(f"⚠️ Non-RoBERTa feature dimension ({node_feature_dim})! Forcing rebuild...")
                        force_rebuild = True
                    else:
                        print("✅ Graph format is compatible")
                        return vocab_data
                else:
                    print("⚠️ Graph has no node features! Forcing rebuild...")
                    force_rebuild = True
            else:
                print("⚠️ Invalid graph format! Forcing rebuild...")
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
            node_feature_dim=768,  # RoBERTa dimension
            use_roberta_init=False
        )
        
        # Save the new graph
        vocab_graph = {
            0: graph,
            1: token2idx,
            'metadata': {
                'num_texts': len(texts),
                'num_tokens': len(token2idx),
                'node_feature_dim': 768,
                'num_edges': graph.edge_index.size(1) if graph.edge_index is not None else 0
            }
        }
        
        with open(file_path, 'wb') as fout:
            pickle.dump(vocab_graph, fout)
        
        print(f"✅ Built and saved new graph: {len(token2idx)} tokens, 768D features")
        return vocab_graph


def load_vocab_graphs_with_filtering(file_path, texts, force_rebuild=False, edge_threshold=0.0, window_size=10):
    """Load or build vocabulary graphs with configurable window size and edge filtering"""
    if file_exists(file_path) and not force_rebuild:
        try:
            print(f"Loading existing vocabulary graph from {file_path}")
            vocab_data = pickle.load(open(file_path, 'rb'))
            
            if isinstance(vocab_data, dict) and 0 in vocab_data and 1 in vocab_data:
                graph, token2idx = vocab_data[0], vocab_data[1]
                
                # Check if metadata contains window_size and it matches
                metadata = vocab_data.get('metadata', {})
                stored_window_size = metadata.get('window_size', None)
                stored_edge_threshold = metadata.get('edge_threshold', None)
                
                if (stored_window_size is not None and stored_window_size != window_size) or \
                   (stored_edge_threshold is not None and stored_edge_threshold != edge_threshold):
                    print(f"Window size or edge threshold mismatch. Stored: win={stored_window_size}, thresh={stored_edge_threshold}. "
                          f"Requested: win={window_size}, thresh={edge_threshold}. Forcing rebuild...")
                    force_rebuild = True
                else:
                    # Apply edge filtering to loaded graph if threshold > 0
                    if edge_threshold > 0.0:
                        print(f"Applying edge filtering with threshold {edge_threshold}")
                        graph = filter_edges_by_threshold(graph, edge_threshold)
                        vocab_data[0] = graph
                    
                    print(f"✅ Loaded graph with window_size={stored_window_size or 'unknown'}, "
                          f"edge_threshold={stored_edge_threshold or 'unknown'}")
                    return vocab_data
                
        except Exception as e:
            print(f"Error loading graph: {e}. Forcing rebuild...")
            force_rebuild = True
    
    if force_rebuild or not file_exists(file_path):
        print(f"Building new vocabulary graph for {len(texts)} texts with window_size={window_size}")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Build the graph with configurable window size and edge filtering
        graph, token2idx = build_pyg_graph_efficient_with_filtering(
            texts, 
            tokenizer=tokenizer, 
            window_size=window_size,  # Use configurable window size
            min_count=1,
            node_feature_dim=768,
            use_roberta_init=False,
            edge_threshold=edge_threshold
        )
        
        vocab_graph = {
            0: graph,
            1: token2idx,
            'metadata': {
                'num_texts': len(texts),
                'num_tokens': len(token2idx),
                'node_feature_dim': 768,
                'num_edges': graph.edge_index.size(1) if graph.edge_index is not None else 0,
                'edge_threshold': edge_threshold,
                'window_size': window_size,
                'creation_timestamp': str(torch.utils.data.default_collate.__module__)  # Simple timestamp
            }
        }
        
        with open(file_path, 'wb') as fout:
            pickle.dump(vocab_graph, fout)
        
        print(f"✅ Built and saved new graph: {len(token2idx)} tokens, {graph.edge_index.size(1)} edges, "
              f"window_size={window_size}, edge_threshold={edge_threshold}")
        return vocab_graph


def file_exists(file_path):
    """Check if file exists"""
    return os.path.exists(file_path)


def load_articles(obj):
    """Load news articles for training and testing"""
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))   

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    
    return x_train, x_test, y_train, y_test


def analyze_window_size_effects(texts, tokenizer, window_sizes=[5, 10, 15, 20]):
    """Analyze the effect of different window sizes on graph structure"""
    print("Analyzing window size effects...")
    
    results = {}
    
    for window_size in window_sizes:
        print(f"\nTesting window_size={window_size}")
        
        # Build graph with current window size
        graph, token2idx = build_pyg_graph_efficient_with_filtering(
            texts[:100],  # Use subset for speed
            tokenizer=tokenizer,
            window_size=window_size,
            min_count=1,
            node_feature_dim=768,
            use_roberta_init=False,
            edge_threshold=0.0
        )
        
        # Compute graph statistics
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        # Compute edge weight statistics
        if graph.edge_attr is not None and graph.edge_attr.size(0) > 0:
            edge_weights = graph.edge_attr
            weight_stats = {
                'mean': edge_weights.mean().item(),
                'std': edge_weights.std().item(),
                'min': edge_weights.min().item(),
                'max': edge_weights.max().item(),
                'median': edge_weights.median().item()
            }
        else:
            weight_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        results[window_size] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'edge_density': edge_density,
            'avg_degree': avg_degree,
            'edge_weight_stats': weight_stats
        }
        
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"  Density: {edge_density:.4f}, Avg Degree: {avg_degree:.2f}")
        print(f"  Edge weights - Mean: {weight_stats['mean']:.4f}, Std: {weight_stats['std']:.4f}")
    
    # Print comparison
    print("\n" + "="*60)
    print("WINDOW SIZE COMPARISON")
    print("="*60)
    print(f"{'Window':<8} {'Nodes':<8} {'Edges':<8} {'Density':<10} {'AvgDeg':<8} {'AvgWeight':<10}")
    print("-" * 60)
    
    for window_size in window_sizes:
        r = results[window_size]
        print(f"{window_size:<8} {r['num_nodes']:<8} {r['num_edges']:<8} "
              f"{r['edge_density']:<10.4f} {r['avg_degree']:<8.2f} {r['edge_weight_stats']['mean']:<10.4f}")
    
    print("\nWindow Size Guidelines:")
    print("- Small windows (5-8): Focus on local syntactic relationships")
    print("- Medium windows (10-15): Balance between local and broader semantic relationships")
    print("- Large windows (20+): Capture broader topical relationships but may introduce noise")
    
    return results


def get_optimal_hyperparameters(texts, tokenizer, test_window_sizes=[5, 10, 15, 20], 
                               test_edge_thresholds=[0.0, 0.1, 0.2, 0.3]):
    """
    Quick analysis to suggest optimal hyperparameters based on graph characteristics
    """
    print("Finding optimal hyperparameters...")
    
    recommendations = {}
    
    # Test different combinations on a subset of data
    subset_texts = texts[:50] if len(texts) > 50 else texts
    
    for window_size in test_window_sizes:
        for edge_threshold in test_edge_thresholds:
            try:
                graph, token2idx = build_pyg_graph_efficient_with_filtering(
                    subset_texts,
                    tokenizer=tokenizer,
                    window_size=window_size,
                    min_count=1,
                    node_feature_dim=768,
                    use_roberta_init=False,
                    edge_threshold=edge_threshold
                )
                
                # Compute quality metrics
                num_nodes = graph.x.size(0)
                num_edges = graph.edge_index.size(1)
                
                if num_edges > 0:
                    connectivity = num_edges / max(1, num_nodes)
                    edge_density = num_edges / max(1, num_nodes * (num_nodes - 1) / 2)
                    
                    # Simple quality score: balance connectivity and sparsity
                    quality_score = connectivity * (1 - edge_density) if edge_density < 1 else 0
                else:
                    quality_score = 0
                
                recommendations[(window_size, edge_threshold)] = {
                    'nodes': num_nodes,
                    'edges': num_edges,
                    'connectivity': connectivity if num_edges > 0 else 0,
                    'density': edge_density if num_edges > 0 else 0,
                    'quality_score': quality_score
                }
                
            except Exception as e:
                print(f"Failed for window_size={window_size}, edge_threshold={edge_threshold}: {e}")
    
    # Find best configuration
    if recommendations:
        best_config = max(recommendations.keys(), key=lambda k: recommendations[k]['quality_score'])
        best_window, best_threshold = best_config
        
        print(f"\nRecommended hyperparameters:")
        print(f"  Window Size: {best_window}")
        print(f"  Edge Threshold: {best_threshold}")
        print(f"  Quality Score: {recommendations[best_config]['quality_score']:.4f}")
        print(f"  Resulting graph: {recommendations[best_config]['nodes']} nodes, {recommendations[best_config]['edges']} edges")
        
        return best_window, best_threshold, recommendations
    else:
        print("Could not determine optimal hyperparameters")
        return 10, 0.1, {}


def analyze_edge_threshold_effects(texts, tokenizer, window_size=10, edge_thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """Analyze the effect of different edge thresholds on graph structure"""
    print(f"Analyzing edge threshold effects with window_size={window_size}...")
    
    results = {}
    
    # First build the base graph without filtering
    base_graph, token2idx = build_pyg_graph_efficient_with_filtering(
        texts[:100],  # Use subset for speed
        tokenizer=tokenizer,
        window_size=window_size,
        min_count=1,
        node_feature_dim=768,
        use_roberta_init=False,
        edge_threshold=0.0  # No filtering for base graph
    )
    
    original_edges = base_graph.edge_index.size(1)
    print(f"Base graph: {base_graph.x.size(0)} nodes, {original_edges} edges")
    
    for threshold in edge_thresholds:
        print(f"\nTesting edge_threshold={threshold}")
        
        # Apply filtering to base graph
        if threshold > 0.0:
            filtered_graph = filter_edges_by_threshold(base_graph, threshold)
        else:
            filtered_graph = base_graph
        
        # Compute graph statistics
        num_nodes = filtered_graph.x.size(0)
        num_edges = filtered_graph.edge_index.size(1)
        edge_retention_rate = num_edges / original_edges if original_edges > 0 else 0
        edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        # Compute edge weight statistics for remaining edges
        if filtered_graph.edge_attr is not None and filtered_graph.edge_attr.size(0) > 0:
            edge_weights = filtered_graph.edge_attr
            weight_stats = {
                'mean': edge_weights.mean().item(),
                'std': edge_weights.std().item(),
                'min': edge_weights.min().item(),
                'max': edge_weights.max().item(),
                'median': edge_weights.median().item()
            }
        else:
            weight_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        results[threshold] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'edge_retention_rate': edge_retention_rate,
            'edge_density': edge_density,
            'avg_degree': avg_degree,
            'edge_weight_stats': weight_stats
        }
        
        print(f"  Edges: {num_edges} ({edge_retention_rate*100:.1f}% retained)")
        print(f"  Density: {edge_density:.4f}, Avg Degree: {avg_degree:.2f}")
        print(f"  Remaining edge weights - Mean: {weight_stats['mean']:.4f}")
    
    # Print comparison
    print("\n" + "="*70)
    print("EDGE THRESHOLD COMPARISON")
    print("="*70)
    print(f"{'Threshold':<10} {'Edges':<8} {'Retained%':<10} {'Density':<10} {'AvgDeg':<8} {'MinWeight':<10}")
    print("-" * 70)
    
    for threshold in edge_thresholds:
        r = results[threshold]
        print(f"{threshold:<10} {r['num_edges']:<8} {r['edge_retention_rate']*100:<10.1f} "
              f"{r['edge_density']:<10.4f} {r['avg_degree']:<8.2f} {r['edge_weight_stats']['min']:<10.4f}")
    
    print("\nEdge Threshold Guidelines:")
    print("- 0.0: Keep all edges (dense graph, includes weak connections)")
    print("- 0.1-0.2: Remove weak connections (balanced approach)")
    print("- 0.3-0.4: Keep only moderate to strong connections")
    print("- 0.5+: Keep only very strong connections (sparse graph)")
    
    return results


def create_hyperparameter_analysis_report(texts, tokenizer, output_file='hyperparameter_analysis.txt'):
    """Create comprehensive hyperparameter analysis report"""
    print("Creating comprehensive hyperparameter analysis report...")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Window size analysis
        f.write("1. WINDOW SIZE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        window_results = analyze_window_size_effects(texts, tokenizer)
        
        f.write("\nWindow Size Results:\n")
        for window_size, stats in window_results.items():
            f.write(f"Window {window_size}: {stats['num_nodes']} nodes, {stats['num_edges']} edges, "
                   f"density={stats['edge_density']:.4f}, avg_weight={stats['edge_weight_stats']['mean']:.4f}\n")
        
        # Edge threshold analysis
        f.write("\n\n2. EDGE THRESHOLD ANALYSIS\n")
        f.write("-" * 40 + "\n")
        edge_results = analyze_edge_threshold_effects(texts, tokenizer)
        
        f.write("\nEdge Threshold Results:\n")
        for threshold, stats in edge_results.items():
            f.write(f"Threshold {threshold}: {stats['num_edges']} edges "
                   f"({stats['edge_retention_rate']*100:.1f}% retained), "
                   f"density={stats['edge_density']:.4f}\n")
        
        # Optimal hyperparameters
        f.write("\n\n3. RECOMMENDED HYPERPARAMETERS\n")
        f.write("-" * 40 + "\n")
        optimal_window, optimal_threshold, analysis = get_optimal_hyperparameters(texts, tokenizer)
        
        f.write(f"Recommended Window Size: {optimal_window}\n")
        f.write(f"Recommended Edge Threshold: {optimal_threshold}\n")
        f.write(f"Expected graph: {analysis.get((optimal_window, optimal_threshold), {}).get('nodes', 'N/A')} nodes, "
               f"{analysis.get((optimal_window, optimal_threshold), {}).get('edges', 'N/A')} edges\n")
        
        # Guidelines
        f.write("\n\n4. HYPERPARAMETER GUIDELINES\n")
        f.write("-" * 40 + "\n")
        f.write("Window Size Guidelines:\n")
        f.write("- 5-8: Local syntactic relationships, good for fine-grained analysis\n")
        f.write("- 10-15: Balanced local and semantic relationships (RECOMMENDED)\n")
        f.write("- 20+: Broad topical relationships, may introduce noise\n\n")
        
        f.write("Edge Threshold Guidelines:\n")
        f.write("- 0.0: Keep all edges, maximum information but potential noise\n")
        f.write("- 0.1-0.2: Remove weak connections, balanced approach (RECOMMENDED)\n")
        f.write("- 0.3+: Keep only strong connections, sparse but focused\n\n")
        
        f.write("Task-Specific Recommendations:\n")
        f.write("- Fake News Detection: window_size=10-15, edge_threshold=0.1-0.2\n")
        f.write("- Short Text Analysis: window_size=5-8, edge_threshold=0.0-0.1\n")
        f.write("- Long Document Analysis: window_size=15-20, edge_threshold=0.2-0.3\n")
    
    print(f"Hyperparameter analysis report saved to: {output_file}")
    return output_file


def validate_hyperparameters(window_size, edge_threshold):
    """Validate hyperparameter values and provide warnings"""
    warnings = []
    
    if window_size < 3:
        warnings.append(f"Very small window_size ({window_size}) may miss important relationships")
    elif window_size > 25:
        warnings.append(f"Very large window_size ({window_size}) may introduce noise")
    
    if edge_threshold < 0:
        warnings.append(f"Negative edge_threshold ({edge_threshold}) is invalid")
    elif edge_threshold > 0.8:
        warnings.append(f"Very high edge_threshold ({edge_threshold}) may result in too few edges")
    
    if edge_threshold > 0.5 and window_size < 8:
        warnings.append("Combination of high edge_threshold and small window_size may result in very sparse graphs")
    
    if warnings:
        print("⚠️ Hyperparameter Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Hyperparameters look good!")
    
    return len(warnings) == 0