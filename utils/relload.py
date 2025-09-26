import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import spacy
from collections import defaultdict
from typing import List, Tuple, Dict, Any


class DependencyProcessor:
    """Processes dependency parses for the relationship model"""
    
    def __init__(self, spacy_model='en_core_web_sm'):
        self.nlp = spacy.load(spacy_model)
        self.dep_type_vocab = self._build_dep_vocab()
        self.num_dep_types = len(self.dep_type_vocab)
        
    def _build_dep_vocab(self):
        """Build vocabulary of dependency types"""
        # Common Universal Dependencies
        dep_types = [
            'ROOT', 'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp',
            'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'advmod',
            'discourse', 'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod',
            'acl', 'amod', 'det', 'clf', 'case', 'conj', 'cc', 'fixed',
            'flat', 'compound', 'list', 'parataxis', 'orphan', 'goeswith',
            'reparandum', 'punct', 'dep', 'nsubjpass', 'auxpass', 'attr',
            'dobj', 'pobj', 'prep', 'relcl', 'quantmod', 'partmod', 'infmod',
            'pcomp', 'preconj', 'prt', 'neg', 'possessive', 'npadvmod'
        ]
        return {dep: idx for idx, dep in enumerate(dep_types)}
    
    def parse_sentence(self, text):
        """Parse a sentence and extract dependency information"""
        doc = self.nlp(text)
        
        # Build token to index mapping for transformer alignment
        tokens = [token.text for token in doc]
        
        # Extract dependency edges
        dependencies = []
        for token in doc:
            if token.dep_ != "ROOT":
                gov_idx = token.head.i
                dep_idx = token.i
                dep_type = token.dep_
                
                # Handle unknown dependency types
                if dep_type not in self.dep_type_vocab:
                    dep_type = 'dep'  # generic dependency
                    
                dependencies.append((gov_idx, dep_idx, dep_type))
        
        return tokens, dependencies
    
    def create_meta_relationships(self, dependencies):
        """Create meta-relationships between dependencies that share nodes"""
        meta_rels = []
        
        for i, dep1 in enumerate(dependencies):
            for j, dep2 in enumerate(dependencies[i+1:], i+1):
                # Check if dependencies share a node
                gov1, dep1_idx, _ = dep1
                gov2, dep2_idx, _ = dep2
                
                connection_type = None
                if gov1 == gov2:
                    connection_type = 0  # share governor/head
                elif dep1_idx == dep2_idx:
                    connection_type = 1  # share dependent
                elif gov1 == dep2_idx or dep1_idx == gov2:
                    connection_type = 2  # chain connection
                
                if connection_type is not None:
                    meta_rels.append((i, j, connection_type))
        
        return meta_rels


def align_tokens_with_spacy(tokenizer_tokens, spacy_tokens, dependencies):
    """Align tokenizer output with spaCy tokens for dependency indices"""
    # This is a simplified version - in practice you'd need more robust alignment
    
    # Create mapping from spacy token positions to tokenizer positions
    spacy_to_tokenizer = {}
    tokenizer_idx = 0
    
    for spacy_idx, spacy_token in enumerate(spacy_tokens):
        # Find corresponding tokenizer tokens
        start_idx = tokenizer_idx
        matched_len = 0
        
        while matched_len < len(spacy_token) and tokenizer_idx < len(tokenizer_tokens):
            matched_len += len(tokenizer_tokens[tokenizer_idx].replace('##', ''))
            tokenizer_idx += 1
        
        # Map spacy index to tokenizer index range
        spacy_to_tokenizer[spacy_idx] = (start_idx, tokenizer_idx - 1)
    
    # Convert dependency indices
    aligned_deps = []
    for gov, dep, dep_type in dependencies:
        if gov in spacy_to_tokenizer and dep in spacy_to_tokenizer:
            # Use first subtoken index for simplicity
            gov_tok = spacy_to_tokenizer[gov][0]
            dep_tok = spacy_to_tokenizer[dep][0]
            aligned_deps.append((gov_tok, dep_tok, dep_type))
    
    return aligned_deps


def process_batch_for_model(texts, labels, tokenizer, dep_processor, max_length=512, 
                           max_deps=50, max_meta_rels=100):
    """Process a batch of texts for the relationship model"""
    
    batch_data = {
        'input_ids': [],
        'attention_mask': [],
        'dependency_edges': [],
        'dependency_types': [],
        'meta_edges': [],
        'meta_connection_types': [],
        'labels': []
    }
    
    for text, label in zip(texts, labels):
        # Tokenize with transformer tokenizer
        encoded = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Parse with spaCy
        spacy_tokens, dependencies = dep_processor.parse_sentence(text)
        
        # Align tokens (simplified - assumes tokenizer produces similar tokens)
        # In practice, you'd need proper subword alignment
        
        # Prepare dependency data
        dep_edges = []
        dep_types = []
        
        for gov, dep, dep_type in dependencies[:max_deps]:
            # Ensure indices are within bounds
            if gov < max_length and dep < max_length:
                dep_edges.append([gov, dep])
                dep_type_idx = dep_processor.dep_type_vocab.get(dep_type, 0)
                dep_types.append(dep_type_idx)
        
        # Pad if necessary
        num_deps = len(dep_edges)
        if num_deps < max_deps:
            dep_edges.extend([[-1, -1]] * (max_deps - num_deps))
            dep_types.extend([0] * (max_deps - num_deps))
        
        # Create meta-relationships
        meta_rels = dep_processor.create_meta_relationships(dependencies[:num_deps])
        
        meta_edges = []
        meta_types = []
        
        for rel1_idx, rel2_idx, conn_type in meta_rels[:max_meta_rels]:
            if rel1_idx < max_deps and rel2_idx < max_deps:
                meta_edges.append([rel1_idx, rel2_idx])
                meta_types.append(conn_type)
        
        # Pad meta relationships
        num_meta = len(meta_edges)
        if num_meta < max_meta_rels:
            meta_edges.extend([[-1, -1]] * (max_meta_rels - num_meta))
            meta_types.extend([0] * (max_meta_rels - num_meta))
        
        # Convert to tensors
        dep_edges_tensor = torch.tensor(dep_edges)
        dep_types_tensor = torch.nn.functional.one_hot(
            torch.tensor(dep_types),
            num_classes=dep_processor.num_dep_types
        ).float()
        
        meta_edges_tensor = torch.tensor(meta_edges)
        meta_types_tensor = torch.nn.functional.one_hot(
            torch.tensor(meta_types),
            num_classes=3  # 3 connection types
        ).float()
        
        # Add to batch
        batch_data['input_ids'].append(encoded['input_ids'].squeeze(0))
        batch_data['attention_mask'].append(encoded['attention_mask'].squeeze(0))
        batch_data['dependency_edges'].append(dep_edges_tensor)
        batch_data['dependency_types'].append(dep_types_tensor)
        batch_data['meta_edges'].append(meta_edges_tensor)
        batch_data['meta_connection_types'].append(meta_types_tensor)
        batch_data['labels'].append(label)
    
    # Stack all tensors
    for key in batch_data:
        if key != 'labels':
            batch_data[key] = torch.stack(batch_data[key])
        else:
            batch_data[key] = torch.tensor(batch_data[key])
    
    return batch_data


def load_articles(obj, tokenizer_name='bert-base-uncased'):
    """Load articles and prepare them for relationship model"""
    print('Dataset: ', obj)
    print("loading news articles")

    # Load original data
    train_dict = pickle.load(open('data/news_articles/' + obj + '_train.pkl', 'rb'))
    test_dict = pickle.load(open('data/news_articles/' + obj + '_test.pkl', 'rb'))
    restyle_dict = pickle.load(open('data/adversarial_test/' + obj+ '_test_adv_A.pkl', 'rb'))

    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    x_test_res = restyle_dict['news']
    
    # Initialize processors
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dep_processor = DependencyProcessor()
    
    print("Processing dependency parses...")
    
    # Process data for relationship model
    train_data = process_batch_for_model(x_train, y_train, tokenizer, dep_processor)
    test_data = process_batch_for_model(x_test, y_test, tokenizer, dep_processor)
    
    # Process adversarial test data (with same labels as original test)
    test_adv_data = process_batch_for_model(x_test_res, y_test, tokenizer, dep_processor)
    
    return train_data, test_data, test_adv_data, dep_processor.num_dep_types


def create_data_loader(data_dict, batch_size=16, shuffle=True):
    """Create PyTorch DataLoader from processed data"""
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(
        data_dict['input_ids'],
        data_dict['attention_mask'],
        data_dict['dependency_edges'],
        data_dict['dependency_types'],
        data_dict['meta_edges'],
        data_dict['meta_connection_types'],
        data_dict['labels']
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Example usage
if __name__ == "__main__":
    # Load and process data
    train_data, test_data, test_adv_data, num_dep_types = load_articles('gossipcop')
    
    # Create data loaders
    train_loader = create_data_loader(train_data, batch_size=16)
    test_loader = create_data_loader(test_data, batch_size=16, shuffle=False)
    test_adv_loader = create_data_loader(test_adv_data, batch_size=16, shuffle=False)
    
    print(f"Number of dependency types: {num_dep_types}")
    print(f"Train samples: {len(train_data['labels'])}")
    print(f"Test samples: {len(test_data['labels'])}")
    
    # Check batch shapes
    for batch in train_loader:
        input_ids, attention_mask, dep_edges, dep_types, meta_edges, meta_types, labels = batch
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Dependency edges shape: {dep_edges.shape}")
        print(f"Dependency types shape: {dep_types.shape}")
        print(f"Meta edges shape: {meta_edges.shape}")
        print(f"Meta connection types shape: {meta_types.shape}")
        print(f"Labels shape: {labels.shape}")
        break