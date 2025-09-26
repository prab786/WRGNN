import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data,Dataset
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
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

DEP_RELATION_WEIGHTS = {
    "ROOT": 1.0, "nsubj": 0.9, "dobj": 0.8, "amod": 0.7,
    "prep": 0.6, "pobj": 0.5, "det": 0.4, "advmod": 0.3, "other": 0.1
}

DEP_RELATION_TYPE = {
    "ROOT": 0, "nsubj":1, "dobj":2, "amod": 3,
    "prep": 4, "pobj": 5, "det": 6, "advmod": 7, "other": 8
}

def construct_word_based_graph(text, save_path=None, tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Constructs a WORD-BASED graph representation for MSynFD model.
    
    Args:
        text (str): Input text.
        tokenizer: BERT tokenizer (used for word span creation).
        spacy_nlp: spaCy NLP model.
        save_path (str): File path to save the graph information.

    Returns:
        torch_geometric.data.Data: Graph representation with edge index.
        list: List of word spans for BERT subword aggregation.
        pd.DataFrame: DataFrame containing graph edges and their dependency relations.
    """
    global graph_counter

    # Process text with spaCy to get word-level dependencies
    doc = spacy_nlp(text)
    spacy_words = [token.text for token in doc]
    spacy_dependencies = [(token.i, token.dep_, token.head.i) for token in doc]  # (word_idx, dep, head_idx)

    # Create word spans for BERT subword aggregation
    word_spans = create_word_spans_for_bert(text, tokenizer)
    
    # Create WORD-BASED dependency graph (not subword-based)
    edges = []
    edge_weights = []
    dependency_relations = []
    
    for word_idx, dep, head_idx in spacy_dependencies:
        # Check if indices are within our word span range
        if word_idx < len(word_spans) and head_idx < len(word_spans):
            edges.append([word_idx, head_idx])
            dependency_relations.append(DEP_RELATION_TYPE.get(dep, DEP_RELATION_TYPE["other"]))
            edge_weights.append(DEP_RELATION_WEIGHTS.get(dep, DEP_RELATION_WEIGHTS["other"]))
    
    # Convert edges to torch tensors
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        edge_type = torch.tensor(dependency_relations, dtype=torch.long)
    else:
        # Create empty tensors if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
        edge_type = torch.empty((0,), dtype=torch.long)

    # Construct the graph data object for torch-geometric
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight, edge_type=edge_type)
    
    # Create DataFrame for logging (optional)
    df = pd.DataFrame({
        "graph_id": [graph_counter] * len(edges),
        "source_word": [edges[i][0] if i < len(edges) else None for i in range(len(edges))],
        "target_word": [edges[i][1] if i < len(edges) else None for i in range(len(edges))],
        "dependency_relation": dependency_relations,
        "edge_weight": edge_weights
    }) if edges else pd.DataFrame()

    graph_counter += 1

    return graph_data, word_spans, df

def create_word_spans_for_bert(text, tokenizer):
    """
    Create word spans that map words to BERT subword token ranges.
    This is what MSynFD uses in its tran_indices.
    
    Args:
        text (str): Input text
        tokenizer: BERT tokenizer
        
    Returns:
        list: List of [start, end] spans for each word in BERT token space
    """
    # Split text into words
    words = text.split()
    
    # Get BERT encoding to understand subword boundaries
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    tokens = encoded.tokens()
    offsets = encoded.offset_mapping
    
    word_spans = []
    current_token_idx = 1  # Skip [CLS] token
    
    for word in words:
        # Find how many BERT tokens this word produces
        word_tokens = tokenizer.tokenize(word)
        
        if len(word_tokens) > 0:
            start_idx = current_token_idx
            end_idx = current_token_idx + len(word_tokens)
            word_spans.append([start_idx, end_idx])
            current_token_idx = end_idx
        
        # Stop if we exceed reasonable length
        if current_token_idx >= 510:  # Leave room for [SEP]
            break
    
    return word_spans

def word_to_adjacency_matrix(graph_data, num_words):
    """
    Convert word-based graph to adjacency matrix for MSynFD with exact dimension matching.
    
    Args:
        graph_data: PyTorch Geometric Data object with word-level edges
        num_words: Number of words (size of adjacency matrix)
        
    Returns:
        torch.Tensor: Adjacency matrix of shape [num_words, num_words]
    """
    if num_words <= 0:
        num_words = 1
        
    adj_matrix = torch.eye(num_words, dtype=torch.float)  # Start with self-loops
    
    if hasattr(graph_data, 'edge_index') and graph_data.edge_index.size(1) > 0:
        edge_index = graph_data.edge_index
        edge_weights = getattr(graph_data, 'edge_weight', torch.ones(edge_index.size(1)))
        
        # Filter edges to be within matrix bounds
        valid_mask = (edge_index[0] < num_words) & (edge_index[1] < num_words)
        valid_edges = edge_index[:, valid_mask]
        valid_weights = edge_weights[valid_mask]
        
        # Set adjacency matrix values
        if valid_edges.size(1) > 0:
            adj_matrix[valid_edges[0], valid_edges[1]] = valid_weights
    
    # Add sequential connections if not already present
    for i in range(num_words - 1):
        if adj_matrix[i, i + 1] == 0:
            adj_matrix[i, i + 1] = 1.0
        if adj_matrix[i + 1, i] == 0:
            adj_matrix[i + 1, i] = 1.0
    
    # Fill remaining positions with high distance values
    for i in range(num_words):
        for j in range(num_words):
            if adj_matrix[i, j] == 0 and i != j:
                adj_matrix[i, j] = 10.0
    
    return adj_matrix
# Updated MSynFD Dataset classes that use word-based graphs

class MSynFDDatasetWordBased(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, language='english'):
        """
        Dataset class for MSynFD model with WORD-BASED graphs
        
        Args:
            texts: List of text strings
            labels: List of labels (integers)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            language: Language for processing ('english' or 'chinese')
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Process text with BERT tokenizer
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_indices = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Create WORD-BASED graph and word spans
        graph_data, tran_indices, _ = construct_word_based_graph(text, tokenizer=self.tokenizer)
        
        # Extract keywords (using simple heuristic)
        keywords = self._extract_keywords(text)
        key_encoded = self.tokenizer(
            keywords,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        key_words = key_encoded['input_ids'].squeeze(0)
        key_mask = key_encoded['attention_mask'].squeeze(0)
        
        # Create entity information (placeholder)
        entity, entity_mask = self._create_entity_info(text)
        
        # Convert graph to adjacency matrix
        root_adj = word_to_adjacency_matrix(graph_data, len(tran_indices))
        
        return {
            'text_info': text,
            'text_indices': text_indices,
            'tran_indices': tran_indices,
            'key_words': key_words,
            'key_mask': key_mask,
            'entity': entity,
            'entity_mask': entity_mask,
            'root_adj': root_adj,
            'label': torch.tensor(label, dtype=torch.long),
            'attention_mask': attention_mask
        }
    
    def _extract_keywords(self, text):
        """Extract keywords from text (simple implementation)"""
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word.lower() not in stop_words and len(word) > 3]
        keywords = keywords[:10] if len(keywords) > 10 else keywords
        return ' '.join(keywords) if keywords else text[:50]
    
    def _create_entity_info(self, text):
        """Create entity information (placeholder implementation)"""
        entity_tokens = self.tokenizer(
            text[:100],
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return entity_tokens['input_ids'].squeeze(0), entity_tokens['attention_mask'].squeeze(0)

def collate_fn_word_based(batch):
    """Custom collate function for word-based MSynFD DataLoader"""
    # Extract individual components
    text_info = [item['text_info'] for item in batch]
    text_indices = torch.stack([item['text_indices'] for item in batch])
    key_words = torch.stack([item['key_words'] for item in batch])
    key_mask = torch.stack([item['key_mask'] for item in batch])
    entity = torch.stack([item['entity'] for item in batch])
    entity_mask = torch.stack([item['entity_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Handle variable-length tran_indices
    tran_indices = [item['tran_indices'] for item in batch]
    
    # Handle variable-length adjacency matrices
    max_seq_len = max(len(item['tran_indices']) for item in batch) if tran_indices else 1
    root_adj = torch.zeros(len(batch), max_seq_len, max_seq_len)
    
    for i, item in enumerate(batch):
        adj = item['root_adj']
        seq_len = min(adj.size(0), max_seq_len)
        root_adj[i, :seq_len, :seq_len] = adj[:seq_len, :seq_len]
    
    return {
        'inputs': (text_info, text_indices, tran_indices, key_words, key_mask, entity, entity_mask, root_adj),
        'labels': labels
    }

# Updated data loading functions
def load_word_based_graphs(file_path, texts):
    """Load or create word-based graphs for MSynFD"""
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    
    idx2graph = {}
    print(f"Creating word-based graphs and saving to {file_path}")
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processing {i}/{len(texts)}")
        
        # Create word-based graph
        graph_data, word_spans, _ = construct_word_based_graph(text[:2000])
        idx2graph[i] = {'graph': graph_data, 'word_spans': word_spans}
    
    # Save to file
    with open(file_path, 'wb') as fout:
        pickle.dump(idx2graph, fout)
    
    return idx2graph

def load_articles_word_based(dataset_name):
    """
    Load articles with word-based graphs for MSynFD
    """
    print('Dataset: ', dataset_name)
    print("Loading news articles with word-based graphs")

    # Load original data
    train_dict = pickle.load(open(f'data/news_articles/{dataset_name}_train.pkl', 'rb'))
    test_dict = pickle.load(open(f'data/news_articles/{dataset_name}_test.pkl', 'rb'))
    restyle_dict = pickle.load(open(f'data/adversarial_test/{dataset_name}_test_adv_A.pkl', 'rb'))

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    x_test_res = restyle_dict['news']

    # Load or create word-based graphs
    x_train_graph = load_word_based_graphs(f'gr/word_based_train_{dataset_name}.pkl', x_train)
    x_test_graph = load_word_based_graphs(f'gr/word_based_test_{dataset_name}.pkl', x_test)
    x_test_res_graph = load_word_based_graphs(f'gr/word_based_test_res_{dataset_name}.pkl', x_test_res)

    return x_train, x_test, x_test_res, y_train, y_test, x_train_graph, x_test_graph, x_test_res_graph

def load_reframing_word_based(dataset_name):
    """
    Load reframing data with word-based graphs for MSynFD
    """
    print("Loading news augmentations with word-based graphs")
    print('Dataset: ', dataset_name)

    # Load reframing data
    restyle_dict_train1_1 = pickle.load(open(f'data/reframings/{dataset_name}_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open(f'data/reframings/{dataset_name}_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open(f'data/reframings/{dataset_name}_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open(f'data/reframings/{dataset_name}_train_sensational.pkl', 'rb'))
    
    finegrain_dict1 = pickle.load(open(f'data/veracity_attributions/{dataset_name}_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(open(f'data/veracity_attributions/{dataset_name}_fake_standards_neutral_sensational.pkl', 'rb'))

    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])
    
    # Load or create word-based graphs for reframing data
    x_train_res1_graph = load_word_based_graphs(f'gr/word_based_train_res1_{dataset_name}.pkl', x_train_res1)
    x_train_res1_2_graph = load_word_based_graphs(f'gr/word_based_train_res1_2_{dataset_name}.pkl', x_train_res1_2)
    x_train_res2_graph = load_word_based_graphs(f'gr/word_based_train_res2_{dataset_name}.pkl', x_train_res2)
    x_train_res2_2_graph = load_word_based_graphs(f'gr/word_based_train_res2_2_{dataset_name}.pkl', x_train_res2_2)

    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

    # Random replacement logic
    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]   
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]  
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]
    
    replace_idx = replace_idx.tolist()
    for idx in replace_idx:
        x_train_res1_graph[idx] = x_train_res1_2_graph[idx]
        x_train_res2_graph[idx] = x_train_res2_2_graph[idx]

    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t, x_train_res1_graph, x_train_res2_graph

# Example usage and testing
if __name__ == "__main__":
    # Test the word-based graph creation
    test_text = "The quick brown fox jumps over the lazy dog."
    
    print("Testing word-based graph creation:")
    print(f"Input text: {test_text}")
    
    graph_data, word_spans, df = construct_word_based_graph(test_text)
    
    print(f"Word spans: {word_spans}")
    print(f"Graph edges: {graph_data.edge_index}")
    print(f"Edge weights: {graph_data.edge_weight}")
    print(f"Edge types: {graph_data.edge_type}")
    
    # Test adjacency matrix conversion
    adj_matrix = word_to_adjacency_matrix(graph_data, len(word_spans))
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Adjacency matrix:\n{adj_matrix}")