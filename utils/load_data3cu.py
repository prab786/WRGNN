import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel
from torch_geometric.data import Data
import spacy
from spacy.tokens import Doc
import os
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer and NLP model
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
nlp = spacy.load('en_core_web_lg')  # Use large model for better embeddings

@dataclass
class GraphConfig:
    """Configuration for graph construction"""
    max_seq_length: int = 512
    max_sentences: int = 20
    semantic_threshold: float = 0.7
    entity_window: int = 3  # sentences
    use_coreference: bool = True
    include_fact_checking: bool = True
    edge_dropout: float = 0.1

class MultiGraphConstructor:
    """Constructs multiple types of graphs from news text"""
    
    def __init__(self, config: GraphConfig = GraphConfig()):
        self.config = config
        self.tokenizer = tokenizer
        self.nlp = nlp
        
        # Compile patterns for fact extraction
        self.stat_patterns = [
            re.compile(r'\d+\.?\d*\s*%'),  # Percentages
            re.compile(r'\d+\s*out\s*of\s*\d+'),  # Ratios
            re.compile(r'(?:increased?|decreased?|rose|fell)\s*(?:by|to)\s*\d+'),  # Changes
            re.compile(r'\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?'),  # Money
        ]
        
        self.claim_indicators = [
            'claims', 'states', 'argues', 'believes', 'says', 'reported',
            'according to', 'alleges', 'suggests', 'indicates'
        ]
        
        self.evidence_indicators = [
            'because', 'since', 'due to', 'evidence', 'study shows',
            'research indicates', 'data reveals', 'statistics show'
        ]
    
    def construct_multi_graph(self, text: str) -> Dict[str, Data]:
        """Construct multiple graph representations of the text"""
        
        # Process text with spaCy
        doc = self.nlp(text[:5000])  # Limit text length for processing
        
        # Tokenize with RoBERTa
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        graphs = {}
        
        # 1. Syntactic dependency graph (improved)
        graphs['syntactic'] = self._construct_syntactic_graph(doc, encoding)
        
        # 2. Semantic similarity graph
        graphs['semantic'] = self._construct_semantic_graph(doc)
        
        # 3. Entity relationship graph
        graphs['entity'] = self._construct_entity_graph(doc)
        
        # 4. Claim-evidence graph for fact-checking
        if self.config.include_fact_checking:
            graphs['claim'] = self._construct_claim_evidence_graph(doc)
        
        # 5. Discourse structure graph
        graphs['discourse'] = self._construct_discourse_graph(doc)
        
        # 6. Temporal graph (for time-sensitive claims)
        graphs['temporal'] = self._construct_temporal_graph(doc)
        
        return graphs
    
    def _construct_syntactic_graph(self, doc, encoding) -> Data:
        """Improved syntactic dependency graph"""
        
        # Get token mappings
        token_to_word = encoding.word_ids()[1:-1]  # Remove CLS and SEP
        num_tokens = len(token_to_word)
        
        edges = []
        edge_attrs = []
        
        # Build dependency edges
        for token in doc:
            if token.dep_ != "ROOT" and token.head != token:
                # Find RoBERTa token indices
                token_idx = self._get_token_position(token.idx, encoding)
                head_idx = self._get_token_position(token.head.idx, encoding)
                
                if token_idx is not None and head_idx is not None:
                    for t_idx in token_idx:
                        for h_idx in head_idx:
                            if t_idx < num_tokens and h_idx < num_tokens:
                                edges.append([t_idx, h_idx])
                                edges.append([h_idx, t_idx])  # Bidirectional
                                
                                # Edge attributes: dependency type encoded
                                dep_weight = self._get_dependency_weight(token.dep_)
                                edge_attrs.extend([dep_weight, dep_weight])
        
        # Add sequential connections with lower weight
        for i in range(num_tokens - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
            edge_attrs.extend([0.3, 0.3])  # Lower weight for sequential
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_tokens
        )
    
    def _construct_semantic_graph(self, doc) -> Data:
        """Graph based on semantic similarity between sentences"""
        
        sentences = list(doc.sents)[:self.config.max_sentences]
        num_sentences = len(sentences)
        
        if num_sentences < 2:
            return Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float),
                num_nodes=max(1, num_sentences)
            )
        
        # Get sentence embeddings
        sentence_vectors = []
        for sent in sentences:
            if sent.vector_norm:
                sentence_vectors.append(sent.vector)
            else:
                # Fallback to average word vectors
                vectors = [token.vector for token in sent if token.has_vector]
                if vectors:
                    sentence_vectors.append(np.mean(vectors, axis=0))
                else:
                    sentence_vectors.append(np.zeros(300))
        
        sentence_vectors = np.array(sentence_vectors)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(sentence_vectors)
        
        edges = []
        edge_attrs = []
        
        for i in range(num_sentences):
            for j in range(i + 1, num_sentences):
                sim = similarities[i, j]
                
                # Connect semantically similar sentences
                if sim > self.config.semantic_threshold:
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_attrs.extend([sim, sim])
                
                # Always connect adjacent sentences with lower weight
                if j == i + 1:
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_attrs.extend([0.5, 0.5])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_sentences,
            x=torch.tensor(sentence_vectors, dtype=torch.float)  # Include embeddings
        )
    
    def _construct_entity_graph(self, doc) -> Data:
        """Graph connecting entities and their relationships"""
        
        entities = [(ent.text, ent.label_, ent.start, ent.end) 
                   for ent in doc.ents]
        
        if len(entities) < 2:
            return Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 3), dtype=torch.float),
                num_nodes=max(1, len(entities))
            )
        
        edges = []
        edge_attrs = []
        
        # Group entities by sentence
        sent_entities = defaultdict(list)
        for i, (text, label, start, end) in enumerate(entities):
            sent_idx = doc[start].sent.start
            sent_entities[sent_idx].append(i)
        
        # Connect entities in same or nearby sentences
        for sent_idx, ent_indices in sent_entities.items():
            # Connect all entities in same sentence
            for i in range(len(ent_indices)):
                for j in range(i + 1, len(ent_indices)):
                    idx1, idx2 = ent_indices[i], ent_indices[j]
                    edges.append([idx1, idx2])
                    edges.append([idx2, idx1])
                    
                    # Edge attributes: [same_sentence, distance, type_similarity]
                    type_sim = 1.0 if entities[idx1][1] == entities[idx2][1] else 0.5
                    edge_attrs.append([1.0, 0.0, type_sim])
                    edge_attrs.append([1.0, 0.0, type_sim])
        
        # Connect entities in nearby sentences
        for sent_idx1, ents1 in sent_entities.items():
            for sent_idx2, ents2 in sent_entities.items():
                if 0 < abs(sent_idx1 - sent_idx2) <= self.config.entity_window:
                    for idx1 in ents1:
                        for idx2 in ents2:
                            edges.append([idx1, idx2])
                            edges.append([idx2, idx1])
                            
                            distance = abs(sent_idx1 - sent_idx2) / self.config.entity_window
                            type_sim = 1.0 if entities[idx1][1] == entities[idx2][1] else 0.5
                            edge_attrs.append([0.0, distance, type_sim])
                            edge_attrs.append([0.0, distance, type_sim])
        
        # Add coreference resolution if available
        if self.config.use_coreference:
            coref_edges = self._add_coreference_edges(doc, entities)
            edges.extend(coref_edges)
            edge_attrs.extend([[1.0, 0.0, 1.0]] * len(coref_edges))
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        
        # Create node features for entities
        entity_features = self._create_entity_features(entities, doc)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(entities),
            x=entity_features
        )
    
    def _construct_claim_evidence_graph(self, doc) -> Data:
        """Graph connecting claims with evidence for fact-checking"""
        
        sentences = list(doc.sents)[:self.config.max_sentences]
        
        # Classify sentences
        claims = []
        evidence = []
        facts = []
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.text.lower()
            
            # Check for statistical facts
            has_stats = any(pattern.search(sent.text) for pattern in self.stat_patterns)
            if has_stats:
                facts.append(i)
            
            # Check for claims
            if any(indicator in sent_lower for indicator in self.claim_indicators):
                claims.append(i)
            
            # Check for evidence
            if any(indicator in sent_lower for indicator in self.evidence_indicators):
                evidence.append(i)
        
        num_nodes = len(sentences)
        edges = []
        edge_attrs = []
        
        # Connect claims to nearby evidence
        for claim_idx in claims:
            for evidence_idx in evidence:
                if abs(claim_idx - evidence_idx) <= 3:  # Within 3 sentences
                    edges.append([claim_idx, evidence_idx])
                    distance = abs(claim_idx - evidence_idx) / 3
                    edge_attrs.append([1.0, distance])  # [is_claim_evidence, distance]
        
        # Connect facts to claims
        for fact_idx in facts:
            for claim_idx in claims:
                if abs(fact_idx - claim_idx) <= 2:
                    edges.append([fact_idx, claim_idx])
                    distance = abs(fact_idx - claim_idx) / 2
                    edge_attrs.append([0.5, distance])
        
        # Connect contradictory facts
        contradictions = self._find_contradictions(sentences, facts)
        for idx1, idx2 in contradictions:
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])
            edge_attrs.append([2.0, 1.0])  # High weight for contradictions
            edge_attrs.append([2.0, 1.0])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        # Create node features indicating sentence type
        node_features = torch.zeros((num_nodes, 3))
        for i in range(num_nodes):
            if i in claims:
                node_features[i, 0] = 1.0
            if i in evidence:
                node_features[i, 1] = 1.0
            if i in facts:
                node_features[i, 2] = 1.0
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
            x=node_features
        )
    
    def _construct_discourse_graph(self, doc) -> Data:
        """Graph based on discourse markers and rhetorical structure"""
        
        sentences = list(doc.sents)[:self.config.max_sentences]
        num_sentences = len(sentences)
        
        if num_sentences < 2:
            return Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float),
                num_nodes=max(1, num_sentences)
            )
        
        discourse_markers = {
            'contrast': ['but', 'however', 'although', 'despite', 'nevertheless'],
            'cause': ['because', 'since', 'as', 'due to'],
            'result': ['therefore', 'thus', 'consequently', 'as a result'],
            'addition': ['moreover', 'furthermore', 'additionally', 'also'],
            'example': ['for example', 'for instance', 'such as'],
            'conclusion': ['in conclusion', 'finally', 'in summary', 'overall']
        }
        
        edges = []
        edge_attrs = []
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.text.lower()
            
            # Check for discourse markers
            for rel_type, markers in discourse_markers.items():
                if any(marker in sent_lower for marker in markers):
                    # Connect to previous sentences
                    for j in range(max(0, i-3), i):
                        edges.append([j, i])
                        edge_attrs.append([hash(rel_type) % 5 / 5])  # Normalized relation type
                    
                    # Connect to next sentence if conclusion
                    if rel_type == 'conclusion' and i < num_sentences - 1:
                        edges.append([i, i + 1])
                        edge_attrs.append([0.8])
        
        # Add paragraph structure (assuming double newlines separate paragraphs)
        text = doc.text
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            # Connect first and last sentence of each paragraph
            # This is simplified - would need better paragraph detection
            pass
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_sentences
        )
    
    def _construct_temporal_graph(self, doc) -> Data:
        """Graph connecting temporal expressions and events"""
        
        # Extract temporal expressions
        temporal_ents = [(ent.text, ent.start, ent.end) 
                        for ent in doc.ents if ent.label_ in ['DATE', 'TIME']]
        
        if len(temporal_ents) < 2:
            return Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float),
                num_nodes=max(1, len(temporal_ents))
            )
        
        edges = []
        edge_attrs = []
        
        # Connect temporal expressions in chronological order
        for i in range(len(temporal_ents)):
            for j in range(i + 1, min(i + 3, len(temporal_ents))):
                edges.append([i, j])
                edges.append([j, i])
                distance = (j - i) / len(temporal_ents)
                edge_attrs.extend([distance, distance])
        
        # Connect temporal expressions that might be contradictory
        # (This would need more sophisticated temporal parsing)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        return Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(temporal_ents)
        )
    
    # Helper methods
    def _get_token_position(self, char_idx: int, encoding) -> Optional[List[int]]:
        """Map character position to token indices"""
        offset_mapping = encoding['offset_mapping'][0]
        token_indices = []
        
        for i, (start, end) in enumerate(offset_mapping):
            if start <= char_idx < end:
                token_indices.append(i - 1)  # Adjust for CLS token
        
        return token_indices if token_indices else None
    
    def _get_dependency_weight(self, dep_type: str) -> float:
        """Get weight for dependency type"""
        dep_weights = {
            'nsubj': 1.0, 'nsubjpass': 1.0, 'dobj': 0.9, 'iobj': 0.9,
            'ROOT': 1.0, 'ccomp': 0.8, 'xcomp': 0.8, 'advcl': 0.7,
            'amod': 0.6, 'advmod': 0.6, 'prep': 0.5, 'pobj': 0.5,
            'det': 0.3, 'cc': 0.4, 'conj': 0.7
        }
        return dep_weights.get(dep_type, 0.4)
    
    def _add_coreference_edges(self, doc, entities) -> List[List[int]]:
        """Add edges for coreference resolution"""
        coref_edges = []
        
        # Simple pronoun resolution
        pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them']
        
        for i, token in enumerate(doc):
            if token.text.lower() in pronouns:
                # Find nearest person entity
                for j, (ent_text, ent_label, start, end) in enumerate(entities):
                    if ent_label == 'PERSON' and abs(i - start) < 10:
                        # This is simplified - real coref resolution is more complex
                        pass
        
        return coref_edges
    
    def _create_entity_features(self, entities, doc) -> torch.Tensor:
        """Create feature vectors for entities"""
        features = []
        
        entity_type_map = {
            'PERSON': 0, 'ORG': 1, 'GPE': 2, 'DATE': 3,
            'MONEY': 4, 'PERCENT': 5, 'TIME': 6
        }
        
        for text, label, start, end in entities:
            # One-hot encode entity type
            type_vector = torch.zeros(7)
            type_idx = entity_type_map.get(label, -1)
            if type_idx >= 0:
                type_vector[type_idx] = 1.0
            
            # Add frequency feature
            freq = doc.text.lower().count(text.lower()) / len(doc)
            
            # Add position feature
            position = start / len(doc)
            
            feature = torch.cat([
                type_vector,
                torch.tensor([freq, position])
            ])
            features.append(feature)
        
        return torch.stack(features) if features else torch.zeros((1, 9))
    
    def _find_contradictions(self, sentences, fact_indices) -> List[Tuple[int, int]]:
        """Find potentially contradictory facts"""
        contradictions = []
        
        for i in fact_indices:
            for j in fact_indices:
                if i < j:
                    sent_i = sentences[i].text
                    sent_j = sentences[j].text
                    
                    # Check for numerical contradictions
                    nums_i = re.findall(r'\d+\.?\d*', sent_i)
                    nums_j = re.findall(r'\d+\.?\d*', sent_j)
                    
                    if nums_i and nums_j:
                        # Check if discussing same entity but different numbers
                        # This is simplified - would need entity recognition
                        if len(set(nums_i) & set(nums_j)) == 0:
                            # Different numbers about potentially same thing
                            contradictions.append((i, j))
        
        return contradictions


class GraphBatchProcessor:
    """Efficiently process and batch multiple graphs"""
    
    def __init__(self, cache_dir: str = 'graph_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.constructor = MultiGraphConstructor()
    
    def process_texts(self, texts: List[str], 
                      dataset_name: str,
                      use_cache: bool = True) -> Dict[int, Dict[str, Data]]:
        """Process multiple texts into graphs with caching"""
        
        cache_file = os.path.join(self.cache_dir, f'{dataset_name}_graphs.pkl')
        
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached graphs from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Constructing graphs for {len(texts)} texts")
        graphs = {}
        
        for idx, text in enumerate(texts):
            if idx % 100 == 0:
                logger.info(f"Processing text {idx}/{len(texts)}")
            
            try:
                graphs[idx] = self.constructor.construct_multi_graph(text)
            except Exception as e:
                logger.error(f"Error processing text {idx}: {e}")
                # Create empty graphs as fallback
                graphs[idx] = self._create_empty_graphs()
        
        if use_cache:
            logger.info(f"Saving graphs to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(graphs, f)
        
        return graphs
    
    def _create_empty_graphs(self) -> Dict[str, Data]:
        """Create empty graphs as fallback"""
        empty_graph = Data(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1), dtype=torch.float),
            num_nodes=1
        )
        
        return {
            'syntactic': empty_graph,
            'semantic': empty_graph,
            'entity': empty_graph,
            'claim': empty_graph,
            'discourse': empty_graph,
            'temporal': empty_graph
        }


def clean_edge_index(edge_index, num_nodes=511):
    """Clean edge index to remove invalid edges"""
    if edge_index.size(1) == 0:
        return edge_index
    
    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    return edge_index[:, valid_mask]


def load_articles(obj):
    """Load articles with improved graph construction"""
    print('Dataset:', obj)
    print("Loading news articles...")
    
    # Load text data
    train_dict = pickle.load(open(f'data/news_articles/{obj}_train.pkl', 'rb'))
    test_dict = pickle.load(open(f'data/news_articles/{obj}_test.pkl', 'rb'))
    restyle_dict = pickle.load(open(f'data/adversarial_test/{obj}_test_adv_A.pkl', 'rb'))
    
    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    x_test_res = restyle_dict['news']
    
    # Process graphs with new constructor
    processor = GraphBatchProcessor()
    
    x_train_graphs = processor.process_texts(x_train, f'{obj}_train')
    x_test_graphs = processor.process_texts(x_test, f'{obj}_test')
    x_test_res_graphs = processor.process_texts(x_test_res, f'{obj}_test_res')
    
    return (x_train, x_test, x_test_res, y_train, y_test,
            x_train_graphs, x_test_graphs, x_test_res_graphs)


def load_reframing(obj):
    """Load reframing data with improved graph construction"""
    print("Loading news augmentations...")
    print('Dataset:', obj)
    
    # Load augmentation data
    restyle_dict_train1_1 = pickle.load(open(f'data/reframings/{obj}_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open(f'data/reframings/{obj}_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open(f'data/reframings/{obj}_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open(f'data/reframings/{obj}_train_sensational.pkl', 'rb'))
    
    finegrain_dict1 = pickle.load(open(f'data/veracity_attributions/{obj}_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(open(f'data/veracity_attributions/{obj}_fake_standards_neutral_sensational.pkl', 'rb'))
    
    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])
    
    # Process graphs
    processor = GraphBatchProcessor()
    
    x_train_res1_graphs = processor.process_texts(x_train_res1.tolist(), f'{obj}_train_res1')
    x_train_res1_2_graphs = processor.process_texts(x_train_res1_2.tolist(), f'{obj}_train_res1_2')
    x_train_res2_graphs = processor.process_texts(x_train_res2.tolist(), f'{obj}_train_res2')
    x_train_res2_2_graphs = processor.process_texts(x_train_res2_2.tolist(), f'{obj}_train_res2_2')
    
    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']
    
    # Random replacement
    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)
    
    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]
    
    for idx in replace_idx:
        x_train_res1_graphs[idx] = x_train_res1_2_graphs[idx]
        x_train_res2_graphs[idx] = x_train_res2_2_graphs[idx]
    
    return (x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t,
           x_train_res1_graphs, x_train_res2_graphs)


class AdaptiveGraphSelector:
   """Selects and combines appropriate graphs based on text characteristics"""
   
   def __init__(self):
       self.domain_keywords = {
           'political': ['election', 'president', 'congress', 'senate', 'policy', 
                        'democrat', 'republican', 'vote', 'campaign', 'bill'],
           'health': ['covid', 'vaccine', 'virus', 'disease', 'treatment', 
                     'doctor', 'patient', 'hospital', 'health', 'medical'],
           'financial': ['stock', 'market', 'economy', 'inflation', 'bank', 
                        'investment', 'dollar', 'trade', 'finance', 'crypto'],
           'scientific': ['study', 'research', 'scientist', 'data', 'experiment',
                         'evidence', 'hypothesis', 'theory', 'finding', 'analysis'],
           'social': ['twitter', 'facebook', 'social media', 'post', 'viral',
                     'trending', 'hashtag', 'share', 'like', 'comment']
       }
   
   def select_graphs(self, text: str, all_graphs: Dict[str, Data]) -> Dict[str, Data]:
       """Select most relevant graphs based on text domain"""
       
       # Detect domain
       text_lower = text.lower()
       domain_scores = {}
       
       for domain, keywords in self.domain_keywords.items():
           score = sum(1 for keyword in keywords if keyword in text_lower)
           domain_scores[domain] = score
       
       # Get top domain
       top_domain = max(domain_scores, key=domain_scores.get) # type: ignore
       
       # Select graphs based on domain
       selected_graphs = {}
       
       # Always include syntactic graph
       selected_graphs['syntactic'] = all_graphs['syntactic']
       
       if top_domain == 'political':
           # Focus on entities and claims
           selected_graphs['entity'] = all_graphs['entity']
           selected_graphs['claim'] = all_graphs['claim']
           selected_graphs['discourse'] = all_graphs['discourse']
       
       elif top_domain == 'health' or top_domain == 'scientific':
           # Focus on evidence and temporal consistency
           selected_graphs['claim'] = all_graphs['claim']
           selected_graphs['temporal'] = all_graphs['temporal']
           selected_graphs['semantic'] = all_graphs['semantic']
       
       elif top_domain == 'financial':
           # Focus on numerical facts and temporal data
           selected_graphs['temporal'] = all_graphs['temporal']
           selected_graphs['entity'] = all_graphs['entity']
           selected_graphs['claim'] = all_graphs['claim']
       
       elif top_domain == 'social':
           # Focus on discourse and semantic patterns
           selected_graphs['discourse'] = all_graphs['discourse']
           selected_graphs['semantic'] = all_graphs['semantic']
           selected_graphs['entity'] = all_graphs['entity']
       
       else:
           # Default: use all graphs
           selected_graphs = all_graphs
       
       return selected_graphs


class GraphAugmentor:
   """Augment graphs for improved robustness"""
   
   def __init__(self, drop_edge_prob: float = 0.1, 
                add_edge_prob: float = 0.05,
                node_mask_prob: float = 0.1):
       self.drop_edge_prob = drop_edge_prob
       self.add_edge_prob = add_edge_prob
       self.node_mask_prob = node_mask_prob
   
   def augment_graph(self, graph: Data, training: bool = True) -> Data:
       """Apply augmentations to graph during training"""
       
       if not training:
           return graph
       
       # Clone the graph
       augmented = Data(
           edge_index=graph.edge_index.clone() if graph.edge_index is not None else None,
           edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None,
           num_nodes=graph.num_nodes,
           x=graph.x.clone() if graph.x is not None else None
       )
       
       # Edge dropping
       if self.drop_edge_prob > 0 and augmented.edge_index.size(1) > 0: # type: ignore
           keep_mask = torch.rand(augmented.edge_index.size(1)) > self.drop_edge_prob # type: ignore
           augmented.edge_index = augmented.edge_index[:, keep_mask] # type: ignore
           if augmented.edge_attr is not None:
               augmented.edge_attr = augmented.edge_attr[keep_mask]
       
       # Edge addition (random edges)
       if self.add_edge_prob > 0 and augmented.num_nodes > 1: # type: ignore
           num_new_edges = int(augmented.edge_index.size(1) * self.add_edge_prob) # type: ignore
           if num_new_edges > 0:
               new_edges = torch.randint(0, augmented.num_nodes, (2, num_new_edges)) # type: ignore
               augmented.edge_index = torch.cat([augmented.edge_index, new_edges], dim=1) # type: ignore
               
               if augmented.edge_attr is not None:
                   # Add random attributes for new edges
                   new_attrs = torch.rand((num_new_edges, augmented.edge_attr.size(1)))
                   augmented.edge_attr = torch.cat([augmented.edge_attr, new_attrs], dim=0)
       
       # Node feature masking
       if self.node_mask_prob > 0 and augmented.x is not None:
           mask = torch.rand(augmented.x.size(0)) > self.node_mask_prob
           augmented.x = augmented.x * mask.unsqueeze(1)
       
       return augmented


class GraphMetrics:
   """Compute metrics for graph quality assessment"""
   
   @staticmethod
   def compute_graph_statistics(graph: Data) -> Dict[str, float]:
       """Compute various graph statistics"""
       
       stats = {}
       
       if graph.edge_index.size(1) == 0: # type: ignore
           return {
               'num_nodes': float(graph.num_nodes if graph.num_nodes is not None else 0),
               'num_edges': 0.0,
               'density': 0.0,
               'avg_degree': 0.0,
               'connected_components': float(graph.num_nodes if graph.num_nodes is not None else 0)
           }
       
       # Convert to NetworkX for analysis
       G = nx.Graph()
       edge_list = graph.edge_index.t().numpy()
       G.add_edges_from(edge_list)
       
       stats['num_nodes'] = graph.num_nodes
       stats['num_edges'] = graph.edge_index.size(1) // 2  # Undirected
       stats['density'] = nx.density(G) if G.number_of_nodes() > 0 else 0
       
       degrees = dict(G.degree())
       stats['avg_degree'] = np.mean(list(degrees.values())) if degrees else 0
       stats['connected_components'] = nx.number_connected_components(G)
       
       # Clustering coefficient
       try:
           stats['clustering_coef'] = nx.average_clustering(G)
       except:
           stats['clustering_coef'] = 0
       
       return stats
   
   @staticmethod
   def validate_graph(graph: Data) -> bool:
       """Validate graph structure"""
       
       if graph.edge_index.size(0) != 2:
           return False
       
       if graph.edge_index.size(1) > 0:
           max_idx = graph.edge_index.max().item()
           if max_idx >= graph.num_nodes:
               return False
       
       if graph.edge_attr is not None:
           if graph.edge_attr.size(0) != graph.edge_index.size(1):
               return False
       
       return True


class ImprovedNewsDataset(torch.utils.data.Dataset):
   """Improved dataset class for news with multiple graphs"""
   
   def __init__(self, texts: List[str], labels: List[int], 
                graphs: Dict[int, Dict[str, Data]], 
                tokenizer, max_len: int = 512,
                graph_selector: Optional[AdaptiveGraphSelector] = None,
                augmentor: Optional[GraphAugmentor] = None,
                training: bool = False):
       
       self.texts = texts
       self.labels = labels
       self.graphs = graphs
       self.tokenizer = tokenizer
       self.max_len = max_len
       self.graph_selector = graph_selector or AdaptiveGraphSelector()
       self.augmentor = augmentor
       self.training = training
   
   def __getitem__(self, item):
       text = self.texts[item]
       label = self.labels[item]
       all_graphs = self.graphs[item]
       
       # Select relevant graphs
       selected_graphs = self.graph_selector.select_graphs(text, all_graphs)
       
       # Apply augmentation if training
       if self.augmentor and self.training:
           for graph_type, graph in selected_graphs.items():
               selected_graphs[graph_type] = self.augmentor.augment_graph(graph, self.training)
       
       # Clean edge indices
       for graph_type, graph in selected_graphs.items():
           if hasattr(graph, 'edge_index'):
               graph.edge_index = clean_edge_index(graph.edge_index, num_nodes=self.max_len-1)
       
       # Tokenize text
       encoding = self.tokenizer.encode_plus(
           text, 
           add_special_tokens=True, 
           max_length=self.max_len,
           pad_to_max_length=True, 
           truncation=True, 
           return_token_type_ids=False, 
           return_attention_mask=True, 
           return_tensors='pt'
       )
       
       return {
           'input_ids': encoding['input_ids'].flatten(),
           'attention_mask': encoding['attention_mask'].flatten(),
           'graphs': selected_graphs,
           'labels': torch.tensor(label, dtype=torch.long)
       }
   
   def __len__(self):
       return len(self.texts)


class GraphCollator:
   """Custom collator for batching graphs"""
   
   def __call__(self, batch):
       """Collate batch of items with multiple graphs"""
       
       # Stack regular tensors
       input_ids = torch.stack([item['input_ids'] for item in batch])
       attention_mask = torch.stack([item['attention_mask'] for item in batch])
       labels = torch.stack([item['labels'] for item in batch])
       
       # Batch graphs by type
       graph_types = batch[0]['graphs'].keys()
       batched_graphs = {}
       
       for graph_type in graph_types:
           graphs_list = [item['graphs'][graph_type] for item in batch]
           # Use PyG's Batch class to batch graphs
           from torch_geometric.data import Batch
           batched_graphs[graph_type] = Batch.from_data_list(graphs_list)
       
       return {
           'input_ids': input_ids,
           'attention_mask': attention_mask,
           'graphs': batched_graphs,
           'labels': labels
       }


def create_improved_dataloaders(x_train, y_train, x_test, y_test,
                               train_graphs, test_graphs,
                               tokenizer, max_len=512, batch_size=4):
   """Create improved data loaders with multi-graph support"""
   
   # Create augmentor for training
   augmentor = GraphAugmentor(drop_edge_prob=0.1, add_edge_prob=0.05)
   
   # Create datasets
   train_dataset = ImprovedNewsDataset(
       texts=x_train,
       labels=y_train,
       graphs=train_graphs,
       tokenizer=tokenizer,
       max_len=max_len,
       augmentor=augmentor,
       training=True
   )
   
   test_dataset = ImprovedNewsDataset(
       texts=x_test,
       labels=y_test,
       graphs=test_graphs,
       tokenizer=tokenizer,
       max_len=max_len,
       training=False
   )
   
   # Create collator
   collator = GraphCollator()
   
   # Create data loaders
   train_loader = torch.utils.data.DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       collate_fn=collator,
       num_workers=2,
       pin_memory=True
   )
   
   test_loader = torch.utils.data.DataLoader(
       test_dataset,
       batch_size=batch_size,
       shuffle=False,
       collate_fn=collator,
       num_workers=2,
       pin_memory=True
   )
   
   return train_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
   # Test the improved graph construction
   sample_text = """
   The president claimed that unemployment has dropped to 3.5% this year. 
   However, independent analysts argue that the real rate is closer to 5.2%. 
   According to the Bureau of Labor Statistics, the official rate was 4.1% last month.
   This contradicts the administration's earlier statement that unemployment was at historic lows.
   """
   
   constructor = MultiGraphConstructor()
   graphs = constructor.construct_multi_graph(sample_text)
   
   print("Generated graphs:")
   for graph_type, graph in graphs.items():
       metrics = GraphMetrics.compute_graph_statistics(graph)
       print(f"\n{graph_type} graph:")
       print(f"  Nodes: {metrics['num_nodes']}")
       print(f"  Edges: {metrics['num_edges']}")
       print(f"  Density: {metrics['density']:.3f}")
       print(f"  Avg degree: {metrics['avg_degree']:.2f}")
   
   # Validate graphs
   for graph_type, graph in graphs.items():
       is_valid = GraphMetrics.validate_graph(graph)
       print(f"{graph_type} graph valid: {is_valid}")