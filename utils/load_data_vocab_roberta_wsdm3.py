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

# Load spacy model for stop word removal (or define a custom set)
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_STOP_WORDS = nlp.Defaults.stop_words
except OSError:
    print("Warning: spacy model not found, using NLTK stop words")
    try:
        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords', quiet=True)
        SPACY_STOP_WORDS = set(stopwords.words('english'))
    except ImportError:
        print("Warning: Neither spacy nor NLTK available, using basic stop word list")
        # Basic stop word list as fallback
        SPACY_STOP_WORDS = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
            'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
            'who', 'its', 'now', 'find', 'could', 'made', 'may', 'down', 'side',
            'did', 'get', 'come', 'made', 'can', 'over', 'new', 'sound', 'take',
            'only', 'little', 'work', 'know', 'place', 'year', 'live', 'me',
            'back', 'give', 'most', 'very', 'good', 'sentence', 'man', 'think',
            'say', 'great', 'where', 'help', 'through', 'much', 'before', 'line',
            'right', 'too', 'mean', 'old', 'any', 'same', 'tell', 'boy', 'follow',
            'came', 'want', 'show', 'also', 'around', 'form', 'three', 'small',
            'set', 'put', 'end', 'why', 'again', 'turn', 'here', 'off', 'went',
            'old', 'number', 'great', 'tell', 'men', 'say', 'small', 'every',
            'found', 'still', 'between', 'name', 'should', 'home', 'big', 'give',
            'air', 'line', 'set', 'own', 'under', 'read', 'last', 'never', 'us',
            'left', 'end', 'along', 'while', 'might', 'next', 'sound', 'below',
            'saw', 'something', 'thought', 'both', 'few', 'those', 'always',
            'looked', 'show', 'large', 'often', 'together', 'asked', 'house',
            'don', 't', 'world', 'going', 'want', 'school', 'important', 'until',
            'form', 'food', 'keep', 'children', 'feet', 'land', 'side', 'without',
            'boy', 'once', 'animal', 'life', 'enough', 'took', 'sometimes',
            'four', 'head', 'above', 'kind', 'began', 'almost', 'live', 'page',
            'got', 'earth', 'need', 'far', 'hand', 'high', 'year', 'mother',
            'light', 'country', 'father', 'let', 'night', 'picture', 'being',
            'study', 'second', 'book', 'carry', 'took', 'science', 'eat', 'room',
            'friend', 'began', 'idea', 'fish', 'mountain', 'north', 'once', 'base',
            'hear', 'horse', 'cut', 'sure', 'watch', 'color', 'face', 'wood',
            'main', 'open', 'seem', 'together', 'next', 'white', 'children',
            'begin', 'got', 'walk', 'example', 'ease', 'paper', 'group', 'always',
            'music', 'those', 'both', 'mark', 'often', 'letter', 'until', 'mile',
            'river', 'car', 'feet', 'care', 'second', 'book', 'carry', 'took',
            'science', 'eat', 'room', 'friend', 'began', 'idea', 'fish', 'mountain',
            'north', 'once', 'base', 'hear', 'horse', 'cut', 'sure', 'watch'
        }

# Add RoBERTa-specific tokens that should be treated as stop words
ROBERTA_STOP_WORDS = SPACY_STOP_WORDS.union({
    'Ġthe', 'Ġa', 'Ġan', 'Ġand', 'Ġor', 'Ġbut', 'Ġin', 'Ġon', 'Ġat', 'Ġto',
    'Ġfor', 'Ġof', 'Ġwith', 'Ġby', 'Ġis', 'Ġare', 'Ġwas', 'Ġwere', 'Ġbe',
    'Ġbeen', 'Ġhave', 'Ġhas', 'Ġhad', 'Ġdo', 'Ġdoes', 'Ġdid', 'Ġwill',
    'Ġwould', 'Ġcould', 'Ġshould', 'Ġmay', 'Ġmight', 'Ġcan', 'Ġthis',
    'Ġthat', 'Ġthese', 'Ġthose', 'Ġit', 'Ġits', 'Ġhe', 'Ġshe', 'Ġthey',
    'Ġwe', 'Ġyou', 'Ġi', '.', ',', '!', '?', ';', ':', '"', "'", '-', '–', '—'
})


def remove_stopwords_from_tokens(tokens, stop_words=None, min_token_length=2):
    """
    Remove stop words and very short tokens from a list of tokens
    
    Args:
        tokens: List of tokens to filter
        stop_words: Set of stop words to remove (default: ROBERTA_STOP_WORDS)
        min_token_length: Minimum token length to keep (default: 2)
    
    Returns:
        List of filtered tokens
    """
    if stop_words is None:
        stop_words = ROBERTA_STOP_WORDS
    
    filtered_tokens = []
    for token in tokens:
        # Convert to lowercase for stop word checking
        token_lower = token.lower()
        
        # Skip if it's a stop word
        if token_lower in stop_words:
            continue
            
        # Skip very short tokens (often punctuation or meaningless)
        if len(token) < min_token_length:
            continue
            
        # Skip pure punctuation tokens
        if token.strip() in '.,!?;:"\'()-[]{}':
            continue
            
        # Skip tokens that are just whitespace or special characters
        if not any(c.isalnum() for c in token):
            continue
            
        filtered_tokens.append(token)
    
    return filtered_tokens


def tokenize_roberta_with_stopword_removal(texts, tokenizer, remove_stopwords=True, min_token_length=2):
    """
    Tokenize texts using RoBERTa tokenizer and optionally remove stop words
    
    Args:
        texts: List of texts to tokenize
        tokenizer: RoBERTa tokenizer
        remove_stopwords: Whether to remove stop words (default: True)
        min_token_length: Minimum token length to keep (default: 2)
    
    Returns:
        List of tokenized documents
    """
    tokenized_docs = []
    
    for text in texts:
        tokens = tokenizer.tokenize(text.lower())
        
        if remove_stopwords:
            tokens = remove_stopwords_from_tokens(tokens, min_token_length=min_token_length)
        
        tokenized_docs.append(tokens)
    
    return tokenized_docs


def build_vocab_and_cooc_with_stopword_removal(tokenized_docs, window_size=10, min_count=0):
    """
    Build vocabulary and co-occurrence statistics with stop words already removed
    """
    vocab_counter = Counter()
    co_occurrence = defaultdict(int)
    total_windows = 0

    print(f"Building vocabulary with window_size={window_size}, min_count={min_count} (stop words removed)")
    
    for doc_idx, tokens in enumerate(tokenized_docs):
        if doc_idx % 1000 == 0:
            print(f"Processing document {doc_idx}/{len(tokenized_docs)}")
            
        # Skip documents with too few meaningful tokens
        if len(tokens) < 2:
            continue
            
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
    print(f"Stop words filtering resulted in cleaner vocabulary graph")

    return vocab, token2idx, vocab_counter, co_occurrence, total_windows


def build_pyg_graph_efficient_with_stopword_removal(texts, tokenizer=tokenizer, window_size=10, min_count=1, 
                                                   node_feature_dim=768, use_roberta_init=False, 
                                                   edge_threshold=0.0, remove_stopwords=True, 
                                                   min_token_length=2):
    """
    Efficient graph building with stop word removal and edge filtering
    
    Args:
        texts: List of input texts
        tokenizer: RoBERTa tokenizer
        window_size: Co-occurrence window size
        min_count: Minimum token count to include in vocabulary
        node_feature_dim: Dimension of node features
        use_roberta_init: Whether to use RoBERTa embeddings for initialization
        edge_threshold: Edge weight threshold for filtering
        remove_stopwords: Whether to remove stop words
        min_token_length: Minimum token length to keep
    
    Returns:
        PyTorch Geometric Data object and token2idx mapping
    """
    print(f"Graph building parameters:")
    print(f"  - Window size: {window_size}")
    print(f"  - Edge threshold: {edge_threshold}")
    print(f"  - Min count: {min_count}")
    print(f"  - Node feature dim: {node_feature_dim}")
    print(f"  - Remove stopwords: {remove_stopwords}")
    print(f"  - Min token length: {min_token_length}")
    
    # Tokenize with stop word removal
    tokenized = tokenize_roberta_with_stopword_removal(
        texts, tokenizer, remove_stopwords=remove_stopwords, min_token_length=min_token_length
    )
    
    # Build vocabulary and co-occurrence statistics
    vocab, token2idx, counter, cooc, total_windows = build_vocab_and_cooc_with_stopword_removal(
        tokenized, window_size=window_size, min_count=min_count
    )
    
    # Compute NPMI edges
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
        print(f"Edge filtering: {original_edges} -> {filtered_edges} edges (kept {filtered_edges/original_edges*100:.1f}%)")
        
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


def get_vocab_subgraph_from_text_with_stopword_removal(text, vocab_graph, token2idx, tokenizer, 
                                                      edge_threshold=0.0, remove_stopwords=True, 
                                                      min_token_length=2):
    """
    Robust subgraph extraction with stop word removal and optional edge filtering
    
    Args:
        text: Input text to extract subgraph for
        vocab_graph: Vocabulary graph (PyTorch Geometric Data)
        token2idx: Token to index mapping
        tokenizer: RoBERTa tokenizer
        edge_threshold: Edge weight threshold for filtering
        remove_stopwords: Whether to remove stop words
        min_token_length: Minimum token length to keep
    
    Returns:
        PyTorch Geometric Data object representing the subgraph
    """
    try:
        if not text or not text.strip():
            raise ValueError("Empty text provided")
        
        if not token2idx:
            raise ValueError("Empty token2idx vocabulary")
        
        # Tokenize the input text with stop word removal
        tokens = tokenizer.tokenize(text.lower())
        
        if remove_stopwords:
            tokens = remove_stopwords_from_tokens(tokens, min_token_length=min_token_length)
        
        if not tokens:
            raise ValueError("No meaningful tokens found in text after stop word removal")
        
        # Map to graph node indices - remove duplicates while preserving order
        seen = set()
        node_indices = []
        matched_tokens = []
        
        for token in tokens:
            if token in token2idx and token not in seen:
                node_indices.append(token2idx[token])
                matched_tokens.append(token)
                seen.add(token)
        
        if not node_indices:
            raise ValueError("None of the meaningful tokens from the input text are in the vocabulary graph")

        #print(f"Matched {len(matched_tokens)} meaningful tokens: {matched_tokens[:10]}..." if len(matched_tokens) > 10 else f"Matched tokens: {matched_tokens}")

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
            original_edges = edge_attr.size(0)
            subgraph_data = filter_edges_by_threshold(subgraph_data, edge_threshold)
            filtered_edges = subgraph_data.edge_attr.size(0) if subgraph_data.edge_attr is not None else 0
            # if original_edges > 0:
            #     print(f"Subgraph edge filtering: {original_edges} -> {filtered_edges} edges")

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


def load_vocab_graphs_with_stopword_removal(file_path, texts, force_rebuild=False, edge_threshold=0.0, 
                                           window_size=10, remove_stopwords=True, min_token_length=2):
    """
    Load or build vocabulary graphs with stop word removal and configurable parameters
    
    Args:
        file_path: Path to save/load the graph
        texts: Training texts for building the graph
        force_rebuild: Whether to force rebuilding even if file exists
        edge_threshold: Edge weight threshold for filtering
        window_size: Co-occurrence window size
        remove_stopwords: Whether to remove stop words
        min_token_length: Minimum token length to keep
    
    Returns:
        Dictionary containing graph, token2idx, and metadata
    """
    if file_exists(file_path) and not force_rebuild:
        try:
            print(f"Loading existing vocabulary graph from {file_path}")
            vocab_data = pickle.load(open(file_path, 'rb'))
            
            if isinstance(vocab_data, dict) and 0 in vocab_data and 1 in vocab_data:
                graph, token2idx = vocab_data[0], vocab_data[1]
                
                # Check if metadata contains parameters and they match
                metadata = vocab_data.get('metadata', {})
                stored_window_size = metadata.get('window_size', None)
                stored_edge_threshold = metadata.get('edge_threshold', None)
                stored_remove_stopwords = metadata.get('remove_stopwords', None)
                stored_min_token_length = metadata.get('min_token_length', None)
                
                params_match = (
                    stored_window_size == window_size and
                    stored_edge_threshold == edge_threshold and
                    stored_remove_stopwords == remove_stopwords and
                    stored_min_token_length == min_token_length
                )
                
                if not params_match:
                    print(f"Parameters mismatch. Stored: win={stored_window_size}, thresh={stored_edge_threshold}, "
                          f"stopwords={stored_remove_stopwords}, min_len={stored_min_token_length}")
                    print(f"Requested: win={window_size}, thresh={edge_threshold}, "
                          f"stopwords={remove_stopwords}, min_len={min_token_length}")
                    print("Forcing rebuild...")
                    force_rebuild = True
                else:
                    # Apply edge filtering to loaded graph if threshold > 0
                    if edge_threshold > 0.0:
                        print(f"Applying edge filtering with threshold {edge_threshold}")
                        graph = filter_edges_by_threshold(graph, edge_threshold)
                        vocab_data[0] = graph
                    
                    print(f"✅ Loaded graph with matching parameters")
                    return vocab_data
                
        except Exception as e:
            print(f"Error loading graph: {e}. Forcing rebuild...")
            force_rebuild = True
    
    if force_rebuild or not file_exists(file_path):
        print(f"Building new vocabulary graph for {len(texts)} texts")
        print(f"Parameters: window_size={window_size}, edge_threshold={edge_threshold}, "
              f"remove_stopwords={remove_stopwords}, min_token_length={min_token_length}")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Build the graph with stop word removal and configurable parameters
        graph, token2idx = build_pyg_graph_efficient_with_stopword_removal(
            texts, 
            tokenizer=tokenizer, 
            window_size=window_size,
            min_count=1,
            node_feature_dim=768,
            use_roberta_init=False,
            edge_threshold=edge_threshold,
            remove_stopwords=remove_stopwords,
            min_token_length=min_token_length
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
                'remove_stopwords': remove_stopwords,
                'min_token_length': min_token_length,
                'creation_timestamp': str(torch.utils.data.default_collate.__module__)
            }
        }
        
        with open(file_path, 'wb') as fout:
            pickle.dump(vocab_graph, fout)
        
        print(f"✅ Built and saved new graph: {len(token2idx)} tokens, {graph.edge_index.size(1)} edges")
        print(f"   Parameters: window_size={window_size}, edge_threshold={edge_threshold}, "
              f"stopwords_removed={remove_stopwords}")
        
        return vocab_graph


# Keep all other existing functions unchanged
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
    """Filter edges based on edge attribute threshold"""
    if graph.edge_attr is None or graph.edge_index is None:
        print("Warning: Graph has no edges or edge attributes")
        return graph
    
    edge_mask = graph.edge_attr >= threshold
    filtered_edge_index = graph.edge_index[:, edge_mask]
    filtered_edge_attr = graph.edge_attr[edge_mask]
    
    return Data(
        x=graph.x,
        edge_index=filtered_edge_index,
        edge_attr=filtered_edge_attr
    )


def initialize_with_roberta_embeddings(token2idx, tokenizer, embedding_dim=768):
    """Initialize node features with RoBERTa embeddings"""
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    roberta_model.eval()
    
    num_tokens = len(token2idx)
    x = torch.zeros(num_tokens, embedding_dim)
    
    tokens = list(token2idx.keys())
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i+batch_size]
            
            for token in batch_tokens:
                try:
                    token_ids = tokenizer.encode(token, add_special_tokens=True, return_tensors='pt')
                    outputs = roberta_model(token_ids)
                    token_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = token_embedding[:embedding_dim]
                    
                except Exception as e:
                    print(f"Warning: Could not get RoBERTa embedding for token '{token}': {e}")
                    vocab_idx = token2idx[token]
                    x[vocab_idx] = torch.randn(embedding_dim) * 0.1
    
    return x


def file_exists(file_path):
    """Check if file exists"""
    return os.path.exists(file_path)


def analyze_stopword_impact(texts, tokenizer, window_size=10):
    """
    Analyze the impact of stop word removal on vocabulary graph structure
    """
    print("Analyzing impact of stop word removal...")
    
    results = {}
    
    for remove_stopwords in [False, True]:
        print(f"\nBuilding graph with remove_stopwords={remove_stopwords}")
        
        graph, token2idx = build_pyg_graph_efficient_with_stopword_removal(
            texts[:100],  # Use subset for speed
            tokenizer=tokenizer,
            window_size=window_size,
            min_count=1,
            node_feature_dim=768,
            use_roberta_init=False,
            edge_threshold=0.0,
            remove_stopwords=remove_stopwords
        )
        
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        
        # Sample some vocabulary
        sample_vocab = list(token2idx.keys())[:20]
        
        results[remove_stopwords] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'edge_density': edge_density,
            'avg_degree': avg_degree,
            'sample_vocab': sample_vocab
        }
    
    # Print comparison
    print("\n" + "="*60)
    print("STOP WORD REMOVAL COMPARISON")
    print("="*60)
    
    for remove_sw, stats in results.items():
        print(f"\nRemove Stop Words: {remove_sw}")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Density: {stats['edge_density']:.4f}")
        print(f"  Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  Sample Vocab: {stats['sample_vocab'][:10]}...")
    
    if False in results and True in results:
        reduction_nodes = (results[False]['num_nodes'] - results[True]['num_nodes']) / results[False]['num_nodes'] * 100
        reduction_edges = (results[False]['num_edges'] - results[True]['num_edges']) / results[False]['num_edges'] * 100
        
        print(f"\nStop word removal effects:")
        print(f"  Node reduction: {reduction_nodes:.1f}%")
        print(f"  Edge reduction: {reduction_edges:.1f}%")
        print(f"  This creates cleaner, more focused vocabulary graphs")
    
    return results


# Add the stopword removal to the existing load_articles and split_news functions
def load_articles(obj):
    """Load news articles for training and testing"""
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))   

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    
    return x_train, x_test, y_train, y_test


def split_news_by_binary_label(train_dict):
    """Split news texts into two lists based on binary labels (0 or 1)"""
    news = train_dict['news']
    labels = train_dict['labels']

    if len(news) != len(labels):
        raise ValueError("The number of news items must match the number of labels.")

    news_0 = [n for n, l in zip(news, labels) if l == 0]
    news_1 = [n for n, l in zip(news, labels) if l == 1]

    return news_0, news_1