import re
from typing import List, Tuple, Dict, Optional, Union
import torch
from transformers import BertTokenizer

def extract_words_from_text(text: str, 
                            remove_punctuation: bool = True,
                            lowercase: bool = True,
                            remove_stopwords: bool = False) -> List[str]:
    """
    Extract individual words from input text.
    
    Parameters:
    -----------
    text : str
        The input text to extract words from
    remove_punctuation : bool, default=True
        Whether to remove punctuation from words
    lowercase : bool, default=True
        Whether to convert all words to lowercase
    remove_stopwords : bool, default=False
        Whether to remove common stopwords
        
    Returns:
    --------
    List[str]
        List of extracted words
    """
    # Optional stopwords list (only used if remove_stopwords=True)
    stopwords = set([
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'will', 'just', 'don', 'should', 'now', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
    ])

    # Preprocess text
    if lowercase:
        text = text.lower()
    
    # Split text into words
    if remove_punctuation:
        # Replace punctuation with spaces and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
    else:
        # Just split on whitespace
        words = text.split()
    
    # Remove stopwords if requested
    if remove_stopwords:
        words = [word for word in words if word not in stopwords]
    
    return words

def align_tokens_to_words(text: str, tokenizer: BertTokenizer) -> Tuple[List[str], Dict[int, int]]:
    """
    Aligns BERT tokens to original words and creates a mapping.
    
    Parameters:
    -----------
    text : str
        The input text
    tokenizer : BertTokenizer
        BERT tokenizer
        
    Returns:
    --------
    Tuple[List[str], Dict[int, int]]
        - List of original words
        - Dictionary mapping token indices to word indices
    """
    # Extract words
    words = extract_words_from_text(text, remove_punctuation=False, lowercase=False)
    
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    
    # Initialize mapping: token index -> word index
    token_to_word_map = {}
    
    # Track current position in words and tokens
    word_idx = 0
    token_idx = 1  # Start at 1 to skip [CLS] token
    
    current_word = words[0] if words else ""
    current_tokens = []
    
    while token_idx < len(token_ids) - 1:  # -1 to exclude [SEP] token
        token = tokenizer.decode([token_ids[token_idx]]).strip()
        
        # Remove '##' prefix from BERT subword tokens
        if token.startswith("##"):
            token = token[2:]
        
        current_tokens.append(token)
        token_to_word_map[token_idx] = word_idx
        
        # Check if we've completed the current word
        completed_token = "".join(current_tokens).lower()
        if current_word.lower().startswith(completed_token):
            if completed_token == current_word.lower():
                # Word complete, move to next word
                word_idx += 1
                if word_idx < len(words):
                    current_word = words[word_idx]
                    current_tokens = []
        
        token_idx += 1
    
    return words, token_to_word_map

def get_graph_node_labels(text: str, 
                         tokenizer: BertTokenizer, 
                         graph_data: object,
                         batch_idx: int = 0) -> Dict[int, str]:
    """
    Get word labels for graph nodes based on input text.
    
    Parameters:
    -----------
    text : str
        The input text for which the graph was created
    tokenizer : BertTokenizer
        BERT tokenizer used for encoding the text
    graph_data : object
        PyTorch Geometric graph data object
    batch_idx : int, default=0
        The batch index to extract nodes for
        
    Returns:
    --------
    Dict[int, str]
        Dictionary mapping node indices to word labels
    """
    # Extract words and create token to word mapping
    words, token_to_word_map = align_tokens_to_words(text, tokenizer)
    
    # Get unique node indices for this batch
    edge_index = graph_data.edge_index.cpu().numpy()
    edge_batch = graph_data.batch.cpu().numpy()
    
    # Filter edges for the specified batch
    batch_mask = (edge_batch == batch_idx)
    batch_edges = edge_index[:, batch_mask]
    
    # Get unique nodes in this batch
    unique_nodes = np.unique(batch_edges)
    
    # Create node to word mapping
    node_labels = {}
    for node_idx in unique_nodes:
        # Map node index to token index (they should be aligned)
        if node_idx in token_to_word_map:
            word_idx = token_to_word_map[node_idx]
            if word_idx < len(words):
                node_labels[node_idx] = words[word_idx]
            else:
                node_labels[node_idx] = f"node_{node_idx}"
        else:
            node_labels[node_idx] = f"node_{node_idx}"
    
    return node_labels

def text_to_graph_visualization(model, text, tokenizer, batch_idx=0, **viz_kwargs):
    """
    Process text and visualize its graph attention in a single function.
    
    Parameters:
    -----------
    model : BertClassifier
        Your BERT classifier model with graph attention
    text : str
        Input text to visualize
    tokenizer : BertTokenizer
        BERT tokenizer
    batch_idx : int, default=0
        Which batch to visualize
    **viz_kwargs : dict
        Additional visualization parameters to pass to visualize_attention
        
    Returns:
    --------
    matplotlib.figure.Figure
        The visualization figure
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Create graph data (this would depend on your specific graph creation function)
    # For demonstration, assuming create_graph_from_text is defined elsewhere
    graph_data = create_graph_from_text(text, tokenizer)
    
    # Run model with attention storage
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, graph_data, store_attention=True)
    
    # Extract words as node labels
    words = extract_words_from_text(text)
    
    # Get more accurate node labels if possible
    try:
        node_labels = get_graph_node_labels(text, tokenizer, graph_data, batch_idx)
        # Convert to list format expected by visualize_attention
        # This assumes node indices are sequential from 0 to n-1
        word_list = [node_labels.get(i, f"node_{i}") for i in range(max(node_labels.keys()) + 1)]
    except:
        # Fall back to simple word list if alignment fails
        word_list = words
    
    # Visualize the attention
    fig = model.visualize_attention(
        words=word_list,
        batch_idx=batch_idx,
        **viz_kwargs
    )
    
    return fig