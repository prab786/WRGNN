import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
from torch.utils.data import DataLoader
class MSynFDDataset(Dataset):
    def __init__(self, texts, labels, graphs, tokenizer, max_length=512, language='english'):
        """
        Dataset class for MSynFD model
        
        Args:
            texts: List of text strings
            labels: List of labels (integers)
            graphs: Dictionary mapping indices to graph data
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            language: Language for processing ('english' or 'chinese')
        """
        self.texts = texts
        self.labels = labels
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        graph = self.graphs[idx] if idx in self.graphs else None
        
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
        
        # Create tran_indices (word-level spans for subword aggregation)
        tran_indices = self._create_word_spans(text, encoded)
        
        # Extract keywords (using simple heuristic - can be improved)
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
        
        # Create entity information (placeholder - can be enhanced with NER)
        entity, entity_mask = self._create_entity_info(text)
        
        # Process graph adjacency matrix
        root_adj = self._process_graph_adj(graph, len(tran_indices))
        
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
    
    def _create_word_spans(self, text, encoded):
        """Create word-level spans for subword aggregation"""
        # Get word boundaries using whitespace tokenization
        words = text.split()
        spans = []
        
        # Map words to subword positions
        current_pos = 1  # Skip [CLS] token
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                start_pos = current_pos
                end_pos = current_pos + len(word_tokens)
                spans.append([start_pos, end_pos])
                current_pos = end_pos
            
            # Break if we exceed max length
            if current_pos >= self.max_length - 1:  # Reserve space for [SEP]
                break
        
        return spans
    
    def _extract_keywords(self, text):
        """Extract keywords from text (simple implementation)"""
        # Simple keyword extraction - can be enhanced with more sophisticated methods
        words = text.split()
        # Filter out common stop words and keep important words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word.lower() not in stop_words and len(word) > 3]
        
        # Take first few keywords
        keywords = keywords[:10] if len(keywords) > 10 else keywords
        return ' '.join(keywords) if keywords else text[:50]  # Fallback to first 50 chars
    
    def _create_entity_info(self, text):
        """Create entity information (placeholder implementation)"""
        # Placeholder - in practice, you'd use NER to extract entities
        entity_tokens = self.tokenizer(
            text[:100],  # Use first 100 chars as entity info
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return entity_tokens['input_ids'].squeeze(0), entity_tokens['attention_mask'].squeeze(0)
    
    def _process_graph_adj(self, graph_data, seq_len):
        """Process graph adjacency matrix"""
        if graph_data is None:
            # Create identity matrix if no graph available
            adj = torch.eye(seq_len, dtype=torch.float)
        else:
            # Convert graph edge_index to adjacency matrix
            adj = torch.zeros(seq_len, seq_len, dtype=torch.float)
            
            if hasattr(graph_data, 'edge_index') and graph_data.edge_index.size(1) > 0:
                edge_index = graph_data.edge_index
                
                # Filter edges within sequence length
                valid_edges = (edge_index[0] < seq_len) & (edge_index[1] < seq_len)
                if valid_edges.any():
                    valid_edge_index = edge_index[:, valid_edges]
                    
                    # Set adjacency matrix values
                    if hasattr(graph_data, 'edge_weight'):
                        edge_weights = graph_data.edge_weight[valid_edges]
                        adj[valid_edge_index[0], valid_edge_index[1]] = edge_weights
                    else:
                        adj[valid_edge_index[0], valid_edge_index[1]] = 1.0
            
            # Add self-loops
            adj.fill_diagonal_(1.0)
        
        return adj


def collate_fn(batch):
    """Custom collate function for DataLoader"""
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
    max_seq_len = max(len(item['tran_indices']) for item in batch)
    root_adj = torch.zeros(len(batch), max_seq_len, max_seq_len)
    
    for i, item in enumerate(batch):
        adj = item['root_adj']
        seq_len = min(adj.size(0), max_seq_len)
        root_adj[i, :seq_len, :seq_len] = adj[:seq_len, :seq_len]
    
    return {
        'inputs': (text_info, text_indices, tran_indices, key_words, key_mask, entity, entity_mask, root_adj),
        'labels': labels
    }


def create_msynfd_dataloaders(x_train, y_train, x_test, y_test, train_graphs, test_graphs, 
                             tokenizer, batch_size=16, language='english'):
    """
    Create DataLoaders for MSynFD model
    
    Args:
        x_train, x_test: Training and testing texts
        y_train, y_test: Training and testing labels
        train_graphs, test_graphs: Graph dictionaries
        tokenizer: BERT tokenizer
        batch_size: Batch size for DataLoader
        language: Language setting
    
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    
    
    train_dataset = MSynFDDataset(
        texts=x_train,
        labels=y_train,
        graphs=train_graphs,
        tokenizer=tokenizer,
        language=language
    )
    
    test_dataset = MSynFDDataset(
        texts=x_test,
        labels=y_test,
        graphs=test_graphs,
        tokenizer=tokenizer,
        language=language
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, test_loader


# Example usage function
def load_msynfd_data(dataset_name, batch_size=12, language='english'):
    """
    Complete data loading function for MSynFD
    
    Args:
        dataset_name: Name of the dataset (e.g., 'politifact')
        batch_size: Batch size for DataLoader
        language: Language setting
    
    Returns:
        train_loader, test_loader, test_res_loader: DataLoader objects
    """
    from transformers import AutoTokenizer
    
    # Load tokenizer
    if language == 'chinese':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load data using existing functions
    x_train, x_test, x_test_res, y_train, y_test, train_graphs, test_graphs, test_res_graphs = load_articles(dataset_name)
    
    # Create datasets and loaders
    train_loader, test_loader = create_msynfd_dataloaders(
        x_train, y_train, x_test, y_test,
        train_graphs, test_graphs,
        tokenizer, batch_size, language
    )
    
    # Create test_res loader
    test_res_dataset = MSynFDDataset(
        texts=x_test_res,
        labels=y_test,  # Assuming same labels as test set
        graphs=test_res_graphs,
        tokenizer=tokenizer,
        language=language
    )
    
    test_res_loader = DataLoader(
        test_res_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, test_loader, test_res_loader