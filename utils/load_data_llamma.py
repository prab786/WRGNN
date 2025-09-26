import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from spacy.tokens import Doc
from transformers import AutoTokenizer, AutoModel

# Load LLaMA tokenizer and model
llama_model_name = "meta-llama/Llama-2-7b"  # Replace with your LLaMA model
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModel.from_pretrained(llama_model_name)

# Custom Whitespace Tokenizer for spaCy
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        spaces = [True] * len(words)  # Each token 'owns' a space
        return Doc(self.vocab, words=words, spaces=spaces)

# Load spaCy and set custom tokenizer
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

# Dependency relation weights
DEP_RELATION_WEIGHTS = {
    "root": 1.0, "nsubj": 0.9, "dobj": 0.8, "amod": 0.7,
    "prep": 0.6, "pobj": 0.5, "det": 0.4, "advmod": 0.3, "other": 0.1
}

def construct_text_graph(text, tokenizer=tokenizer, spacy_nlp=nlp):
    """
    Constructs a graph representation of text using LLaMA tokens and spaCy dependencies.

    Args:
        text (str): Input text.
        tokenizer: LLaMA tokenizer.
        spacy_nlp: spaCy NLP model.

    Returns:
        torch_geometric.data.Data: Graph representation with edge index and weights.
        list: List of token IDs for LLaMA tokens.
    """
    # Process text with spaCy
    doc = spacy_nlp(text)
    spacy_tokens = [token.text for token in doc]
    spacy_dependencies = [(token.text, token.dep_, token.head.text, token.i, token.head.i) for token in doc]

    # Tokenize with LLaMA
    tokenized_output = tokenizer(text, return_tensors="pt", truncation=True)
    llama_token_ids = tokenized_output['input_ids'][0].tolist()

    # Map spaCy tokens to LLaMA tokens
    mapping_spacy_to_llama = []
    llama_tokens = tokenizer.convert_ids_to_tokens(llama_token_ids)
    llama_token_index = 0
    buffer = ""

    for spacy_idx, token in enumerate(spacy_tokens):
        subword_indices = []
        while llama_token_index < len(llama_tokens):
            llama_token = llama_tokens[llama_token_index]
            buffer += llama_token.replace("Ġ", "")  # 'Ġ' indicates word boundaries in tokenization
            subword_indices.append(llama_token_index)
            llama_token_index += 1

            if buffer == token:
                buffer = ""  # Reset buffer
                break

        mapping_spacy_to_llama.append((spacy_idx, subword_indices))

    # Create LLaMA-based dependency graph
    edges = []
    for (token, dep, head, token_idx, head_idx) in spacy_dependencies:
        token_llama_indices = [
            idx for spacy_idx, indices in mapping_spacy_to_llama if spacy_idx == token_idx for idx in indices
        ]
        head_llama_indices = [
            idx for spacy_idx, indices in mapping_spacy_to_llama if spacy_idx == head_idx for idx in indices
        ]

        for token_idx in token_llama_indices:
            for head_idx in head_llama_indices:
                edges.append([token_idx, head_idx])

    # Construct adjacency matrix
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([1.0] * len(edges), dtype=torch.float)  # Uniform edge weights

    # Create torch-geometric Data object
    graph_data = Data(edge_index=edge_index, edge_weight=edge_weight)

    return graph_data, llama_token_ids

# Functions for loading articles and constructing graphs
def load_articles(obj):
    print("Dataset: ", obj)
    print("Loading news articles")

    train_dict = pickle.load(open(f"data/news_articles/{obj}_train.pkl", "rb"))
    test_dict = pickle.load(open(f"data/news_articles/{obj}_test.pkl", "rb"))

    x_train, y_train = train_dict["news"], train_dict["labels"]
    x_test, y_test = test_dict["news"], test_dict["labels"]

    x_train_graph = load_graphs(f"llama_train_{obj}.graph", x_train)
    x_test_graph = load_graphs(f"llama_test_{obj}.graph", x_test)

    return x_train, x_test, y_train, y_test, x_train_graph, x_test_graph

def load_graphs(file_path, texts):
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))

    idx2graph = {}
    with open(file_path, "wb") as fout:
        for i, text in enumerate(texts):
            graph_data, _ = construct_text_graph(text[:600])
            idx2graph[i] = graph_data
        pickle.dump(idx2graph, fout)

    return idx2graph

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
    
    x_train_res1_graph=load_graphs('llamax_train_res1'+obj+'.Rograph',x_train_res1)
    x_train_res1_2_graph=load_graphs('llamax_train_res1_2'+obj+'.Rograph',x_train_res1_2)
    x_train_res2_graph=load_graphs('llamax_train_res2'+obj+'.Rograph',x_train_res2)
    x_train_res2_2_graph=load_graphs('llamax_train_res2_2'+obj+'.Rograph',x_train_res2_2)

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
    
load_articles('politifact')