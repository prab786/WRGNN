import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from utils.load_data3_1_roberta import *
from utils.node import * # type: ignore
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.nn import Sequential as Seq, Linear, ReLU,SiLU,GELU,Tanh,LeakyReLU,SELU
warnings.filterwarnings("ignore")
from torch_geometric.nn import GlobalAttention,AttentionalAggregation
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
parser = argparse.ArgumentParser()
#from torch.nn import GeGLU
parser.add_argument('--dataset_name', default='gossipcop', type=str) #politifact/gossipcop/lun
parser.add_argument('--model_name', default='deepENN_ROBERTA_WR-epoch2_all', type=str)
parser.add_argument('--iters', default=10, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from torch_geometric.data import DataLoader
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)
class GeGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)       

class NewsDatasetAug(DataLoader):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len,graphs,x_train_res1_graph,x_train_res2_graph):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2
        self.graphs=graphs
        self.x_train_res1_graph=x_train_res1_graph
        self.x_train_res2_graph=x_train_res2_graph

    def __getitem__(self, item):
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]
        graph = self.graphs[item]
        graph.num_nodes = self.max_len-1
        graph.edge_index=clean_edge_index(graph.edge_index,num_nodes=graph.num_nodes)

        x_train_res1_graph = self.x_train_res1_graph[item]
        x_train_res1_graph.num_nodes = self.max_len-1
        x_train_res1_graph.edge_index=clean_edge_index(x_train_res1_graph.edge_index,num_nodes=x_train_res1_graph.num_nodes)

        x_train_res2_graph = self.x_train_res2_graph[item]
        x_train_res2_graph.num_nodes = self.max_len-1
        x_train_res2_graph.edge_index=clean_edge_index(x_train_res2_graph.edge_index,num_nodes=x_train_res2_graph.num_nodes)

        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        aug1_encoding = self.tokenizer.encode_plus(aug_text1, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        aug2_encoding = self.tokenizer.encode_plus(aug_text2, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug1_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
            'graph':graph,
            'x_train_res1_graph':x_train_res1_graph,
            'x_train_res2_graph':x_train_res2_graph

            
        }

    def __len__(self):
        return len(self.texts)

class NewsDataset(DataLoader):
    def __init__(self, texts, labels, tokenizer, max_len,graphs):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graphs=graphs

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        graph=self.graphs[item]        
        graph.num_nodes = self.max_len-1
        graph.edge_index=clean_edge_index(graph.edge_index,num_nodes=graph.num_nodes)
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                pad_to_max_length=True, truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')
        

        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'graph':graph ,
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)
    
class AttentionTrackingEdgeEmbedding(nn.Module):
    """Enhanced edge embedding module that creates rich edge representations and tracks attention"""
    def __init__(self, in_channels, out_channels):
        super(AttentionTrackingEdgeEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Separate transformations for source and target nodes
        self.source_proj = Linear(in_channels, in_channels)
        self.target_proj = Linear(in_channels, in_channels)
        
        # Attention computation for edge importance
        self.attention_mlp = Seq(
            Linear(2 * in_channels, in_channels),
            ReLU(),
            Linear(in_channels, 1),  # Single attention score per edge
            torch.nn.Sigmoid()  # Normalize attention to [0,1]
        )
        
        # Edge embedding MLP with GeGLU
        self.edge_mlp = Seq(
            Linear(2 * in_channels, out_channels * 2),
            GeGLU(),
            LayerNorm(out_channels),
        )
        
        # Storage for attention weights during forward pass
        self.last_attention_weights = None
        
    def forward(self, node_embeddings, edge_index):
        """
        Create edge embeddings from node pairs and compute attention weights
        Args:
            node_embeddings: [num_nodes, in_channels]
            edge_index: [2, num_edges]
        Returns:
            edge_embeddings: [num_edges, out_channels]
        """
        source_emb = self.source_proj(node_embeddings[edge_index[0]])
        target_emb = self.target_proj(node_embeddings[edge_index[1]])
        
        # Concatenate source and target embeddings
        edge_features = torch.cat([source_emb, target_emb], dim=-1)
        
        # Compute attention weights for each edge
        attention_weights = self.attention_mlp(edge_features)  # [num_edges, 1]
        
        # Store attention weights for visualization
        self.last_attention_weights = attention_weights.detach().cpu().squeeze()
        
        # Apply attention to edge features
        attended_features = edge_features * attention_weights
        
        # Generate final edge embeddings
        edge_embeddings = self.edge_mlp(attended_features)
        
        return edge_embeddings


class MaskAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)
    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores

class RobertaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
        #Freeze all parameters first
        #for param in self.roberta.parameters():
          #  param.requires_grad = False

        self.dropout = nn.Dropout(p = 0.1)
       
        self.ee=AttentionTrackingEdgeEmbedding(self.roberta.config.hidden_size,self.roberta.config.hidden_size) 
        self.ee1=AttentionTrackingEdgeEmbedding(self.roberta.config.hidden_size,self.roberta.config.hidden_size)       
        self.ee2=AttentionTrackingEdgeEmbedding(self.roberta.config.hidden_size,self.roberta.config.hidden_size) 
       
        self.graph_pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(self.roberta.config.hidden_size, 128),
                    nn.GELU(),
                    nn.Linear(128, 1)
                )
        )
        self.graph_pool1 = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(self.roberta.config.hidden_size, 128),
                    nn.GELU(),
                    nn.Linear(128, 1)
                )
        )
        self.graph_pool2 = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(self.roberta.config.hidden_size, 128),
                    nn.GELU(),
                    nn.Linear(128, 1)
                )
        )
     


       
        self.norm = nn.LayerNorm(self.roberta.config.hidden_size)
        self.norm2= nn.LayerNorm(self.roberta.config.hidden_size)
        self.attention=MaskAttention(self.roberta.config.hidden_size)
        self.fc_out = nn.Linear(self.roberta.config.hidden_size, 2)       
       
        self.fc = nn.Linear(self.roberta.config.hidden_size, 2)
        self.fc2 = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.fc_out2 = nn.Linear(self.roberta.config.hidden_size, n_classes)
        self.act=torch.nn.LeakyReLU()
       

    def forward(self, input_ids, attention_mask,graph_data):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)
        all_hidden_states = outputs.hidden_states

        x1=outputs['last_hidden_state']
        x= x1[:, 1:, :]
       
        x=x.flatten(start_dim=0, end_dim=1)
       
        edge_index = graph_data.edge_index
        #print(graph_data)
        edge_batch = graph_data.batch[edge_index[0]]   
       
        EdgeEmbedding = self.ee(x, edge_index) 
        EdgeEmbedding2 = self.ee1(x, edge_index) 
        EdgeEmbedding3 = self.ee2(x, edge_index)
       
        #pooling
        graph_embeddings = self.graph_pool(EdgeEmbedding, edge_batch)
        graph_embeddings1 = self.graph_pool1(EdgeEmbedding2, edge_batch)
        graph_embeddings2 = self.graph_pool2(EdgeEmbedding3, edge_batch)
       
        rep1 = torch.stack((graph_embeddings, graph_embeddings1,graph_embeddings2), dim=1)
        rep, scores = self.attention(rep1)
        rep = self.norm(rep)
        output = self.fc(rep)
        output_g = self.fc2(rep)
        output3 = self.fc_out(pooled_outputs)        
        output3_p = self.fc_out2(pooled_outputs)
        return  output3_p, output3,output_g,output


def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size,graphs,x_train_res1_graph,x_train_res2_graph):
    ds = NewsDatasetAug(texts = contents, aug_texts1 = contents_aug1, aug_texts2 = contents_aug2, labels = np.array(labels), \
                        fg_label = fg_label, aug_fg1 = aug_fg1, aug_fg2 = aug_fg2, tokenizer=tokenizer, max_len=max_len,graphs=graphs,x_train_res1_graph=x_train_res1_graph,x_train_res2_graph=x_train_res2_graph)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size,graphs):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer=tokenizer, max_len=max_len, graphs=graphs )
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

def create_graph_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer=tokenizer, max_len=max_len) # type: ignore
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    


def train_model(tokenizer, max_len, n_epochs, batch_size, datasetname, iter):

    x_train, x_test, x_test_res, y_train, y_test,x_trian_graph,x_test_graph,x_test_res_graph = load_articles(datasetname)
    x_test_resB, x_test_resC, x_test_resD,x_test_res_graphb,x_test_res_graphc,x_test_res_graphd=load_articles_all(datasetname)
    test_loader = create_eval_loader(x_test, y_test, tokenizer, max_len, batch_size,x_test_graph)
    test_loader_res = create_eval_loader(x_test_res, y_test, tokenizer, max_len, batch_size,x_test_res_graph)
    test_loader_resB = create_eval_loader(x_test_resB, y_test, tokenizer, max_len, batch_size,x_test_res_graphb)
    test_loader_resC = create_eval_loader(x_test_resC, y_test, tokenizer, max_len, batch_size,x_test_res_graphc)
    test_loader_resD = create_eval_loader(x_test_resD, y_test, tokenizer, max_len, batch_size,x_test_res_graphd)
    x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t,x_train_res1_graph,x_train_res2_graph = load_reframing(args.dataset_name)
    train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, max_len, batch_size,x_trian_graph,x_train_res1_graph,x_train_res2_graph)

    model = RobertaClassifier(n_classes = 4).to(device)
    train_losses = []
    train_accs = []
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    total_steps = len(train_loader) * n_epochs
    print("Total training steps:", total_steps)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    for epoch in range(n_epochs):
        model.train()
       
        avg_loss = []
        avg_acc = []
        batch_idx = 0

        for Batch_data in tqdm(train_loader):

            input_ids = Batch_data["input_ids"].to(device)
            attention_mask = Batch_data["attention_mask"].to(device)
            graph = Batch_data["graph"].to(device)          
            targets = Batch_data["labels"].to(device)           
            out_labels, out_labels_bi,og,ogbi = model(input_ids=input_ids, attention_mask=attention_mask,graph_data=graph)
           

            sup_criterion = nn.CrossEntropyLoss()
            sup_loss = sup_criterion(ogbi, targets)
            loss =sup_loss 
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()           
            _, pred = ogbi.max(dim=-1)
            correct = pred.eq(targets).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))


        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f}".format(iter, epoch, train_acc))

        if epoch == n_epochs - 1:
            model.eval()
            y_pred = []
            y_pred_res = []
            y_pred_resb = []
            y_pred_resc = []
            y_pred_resd = []
            y_test = []

            for Batch_data in tqdm(test_loader):
                with torch.no_grad():
                    input_ids = Batch_data["input_ids"].to(device)
                    attention_mask = Batch_data["attention_mask"].to(device)
                    targets = Batch_data["labels"].to(device)
                    graph = Batch_data["graph"].to(device)
                    _, val_out1 ,_,val_outg= model(input_ids=input_ids, attention_mask=attention_mask,graph_data=graph)
                   # val_out=(val_out1+val_outg)/2
                    _, val_pred = val_outg.max(dim=1)

                    y_pred.append(val_pred)
                    y_test.append(targets)

            for Batch_data in tqdm(test_loader_res):
                with torch.no_grad():
                    #print(Batch_data["input_ids"].shape)
                    input_ids_aug = Batch_data["input_ids"].to(device)
                    attention_mask_aug = Batch_data["attention_mask"].to(device)
                    graph_aug = Batch_data["graph"].to(device)
                    _, val_out_aug1 ,_, val_out_augg = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug,graph_data=graph_aug)
                    #val_out_aug=(val_out_aug1+val_out_augg)/2
                    _, val_pred_aug = val_out_augg.max(dim=1)
                    #print(val_out_aug.shape)
                    y_pred_res.append(val_pred_aug)
                    #print(val_pred_aug)
                    #print(y_pred_res)

            for Batch_data in tqdm(test_loader_resB):
                with torch.no_grad():
                    #print(Batch_data["input_ids"].shape)
                    input_ids_aug = Batch_data["input_ids"].to(device)
                    attention_mask_aug = Batch_data["attention_mask"].to(device)
                    graph_aug = Batch_data["graph"].to(device)
                    _, val_out_aug1 ,_, val_out_augg = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug,graph_data=graph_aug)
                    #val_out_aug=(val_out_aug1+val_out_augg)/2
                    _, val_pred_aug = val_out_augg.max(dim=1)
                    #print(val_out_aug.shape)
                    y_pred_resb.append(val_pred_aug)
                    #print(val_pred_aug)
                    #print(y_pred_res)
            for Batch_data in tqdm(test_loader_resC):
                with torch.no_grad():
                    #print(Batch_data["input_ids"].shape)
                    input_ids_aug = Batch_data["input_ids"].to(device)
                    attention_mask_aug = Batch_data["attention_mask"].to(device)
                    graph_aug = Batch_data["graph"].to(device)
                    _, val_out_aug1 ,_, val_out_augg = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug,graph_data=graph_aug)
                    #val_out_aug=(val_out_aug1+val_out_augg)/2
                    _, val_pred_aug = val_out_augg.max(dim=1)
                    #print(val_out_aug.shape)
                    y_pred_resc.append(val_pred_aug)
                    #print(val_pred_aug)
                    #print(y_pred_res)
            for Batch_data in tqdm(test_loader_resD):
                with torch.no_grad():
                    #print(Batch_data["input_ids"].shape)
                    input_ids_aug = Batch_data["input_ids"].to(device)
                    attention_mask_aug = Batch_data["attention_mask"].to(device)
                    graph_aug = Batch_data["graph"].to(device)
                    _, val_out_aug1 ,_, val_out_augg = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug,graph_data=graph_aug)
                    #val_out_aug=(val_out_aug1+val_out_augg)/2
                    _, val_pred_aug = val_out_augg.max(dim=1)
                    #print(val_out_aug.shape)
                    y_pred_resd.append(val_pred_aug)
                    #print(val_pred_aug)
                    #print(y_pred_res)
            
            y_pred = torch.cat(y_pred, dim=0)
            y_test = torch.cat(y_test, dim=0)
            y_pred_res = torch.cat(y_pred_res, dim=0)
            y_pred_resb = torch.cat(y_pred_resb, dim=0)
            y_pred_resc = torch.cat(y_pred_resc, dim=0)
            y_pred_resd = torch.cat(y_pred_resd, dim=0)
            #print(y_pred_res.shape)
            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')


            acc_res = accuracy_score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy())
            precision_res, recall_res, fscore_res, _ = score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy(), average='macro')
            acc_resb = accuracy_score(y_test.detach().cpu().numpy(), y_pred_resb.detach().cpu().numpy())
            precision_resb, recall_resb, fscore_resb, _ = score(y_test.detach().cpu().numpy(), y_pred_resb.detach().cpu().numpy(), average='macro')
            acc_resc = accuracy_score(y_test.detach().cpu().numpy(), y_pred_resc.detach().cpu().numpy())
            precision_resc, recall_resc, fscore_resc, _ = score(y_test.detach().cpu().numpy(), y_pred_resc.detach().cpu().numpy(), average='macro')
            acc_resd = accuracy_score(y_test.detach().cpu().numpy(), y_pred_resd.detach().cpu().numpy())
            precision_resd, recall_resd, fscore_resd, _ = score(y_test.detach().cpu().numpy(), y_pred_resd.detach().cpu().numpy(), average='macro')
            

    torch.save(model.state_dict(), 'checkpoint_new/' + datasetname + '_iter' + str(iter) + '.m')

    print("-----------------End of Iter {:03d}-----------------".format(iter))
    print(['Global Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])

    print("-----------------Restyle-----------------")
    print(['RestyleA Test Accuracy:{:.4f}'.format(acc_res),
        'Precision:{:.4f}'.format(precision_res),
        'Recall:{:.4f}'.format(recall_res),
        'F1:{:.4f}'.format(fscore_res)])
    print(['RestyleB Test Accuracy:{:.4f}'.format(acc_resb),
        'Precision:{:.4f}'.format(precision_resb),
        'Recall:{:.4f}'.format(recall_resb),
        'F1:{:.4f}'.format(fscore_resb)])
    print(['RestyleC Test Accuracy:{:.4f}'.format(acc_resc),
        'Precision:{:.4f}'.format(precision_resc),
        'Recall:{:.4f}'.format(recall_resc),
        'F1:{:.4f}'.format(fscore_resc)])
    print(['RestyleD Test Accuracy:{:.4f}'.format(acc_resd),
        'Precision:{:.4f}'.format(precision_resd),
        'Recall:{:.4f}'.format(recall_resd),
        'F1:{:.4f}'.format(fscore_resd)])
    o={'acc_resb':acc_resb,'precision_resb':precision_resb,'recall_resb':recall_resb,'fscore_resb':fscore_resb,
       'acc_resc':acc_resc,'precision_resc':precision_resc,'recall_resc':recall_resc,'fscore_resc':fscore_resc,
       'acc_resd':acc_resd,'precision_resd':precision_resd,'recall_resd':recall_resd,'fscore_resd':fscore_resd}
    return acc, precision, recall, fscore, acc_res, precision_res, recall_res, fscore_res,o


datasetname=args.dataset_name
batch_size = args.batch_size
max_len = 512
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
n_epochs = args.n_epochs
iterations=args.iters
#epoachs_tune=[5]
test_accs = []
prec_all, rec_all, f1_all = [], [], []
test_accs_res = []
prec_all_res, rec_all_res, f1_all_res = [], [], []
f1_r_all, f1_f_all, f1_r_all_res, f1_f_all_res = [], [], [], []
test_accs_resb = []
prec_all_resb, rec_all_resb, f1_all_resb = [], [], []
test_accs_resc = []
prec_all_resc, rec_all_resc, f1_all_resc = [], [], []
test_accs_resd = []
prec_all_resd, rec_all_resd, f1_all_resd = [], [], []


for iter in range(iterations):
    set_seed(iter)
    acc, prec, recall, f1, \
    acc_res, prec_res, recall_res, f1_res,o = train_model(tokenizer,
                                                max_len,
                                                n_epochs,
                                                batch_size,
                                                datasetname,
                                                iter)

    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)
    test_accs_res.append(acc_res)
    prec_all_res.append(prec_res)
    rec_all_res.append(recall_res)
    f1_all_res.append(f1_res)
    test_accs_resb.append(o['acc_resb'])
    prec_all_resb.append(o['precision_resb'])
    rec_all_resb.append(o['recall_resb'])
    f1_all_resb.append(o['fscore_resb'])
    test_accs_resc.append(o['acc_resc'])
    prec_all_resc.append(o['precision_resc'])
    rec_all_resc.append(o['recall_resc'])
    f1_all_resc.append(o['fscore_resc'])
    test_accs_resd.append(o['acc_resd'])
    prec_all_resd.append(o['precision_resd'])
    rec_all_resd.append(o['recall_resd']) 
    f1_all_resd.append(o['fscore_resd'])  

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))

print("Restyle_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs_res) / iterations, sum(prec_all_res) /iterations, sum(rec_all_res) /iterations, sum(f1_all_res) / iterations))
print("RestyleB_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs_resb) / iterations, sum(prec_all_resb) /iterations, sum(rec_all_resb) /iterations, sum(f1_all_resb) / iterations))
print("RestyleC_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs_resc) / iterations, sum(prec_all_resc) / iterations, sum(rec_all_resc) /iterations, sum(f1_all_resc) / iterations))
print("RestyleD_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs_resd) / iterations, sum(prec_all_resd) / iterations, sum(rec_all_resd) /iterations, sum(f1_all_resd) / iterations))

with open('log_paper2/log_' +  datasetname + '_' + args.model_name + '.' + 'iter' + str(iterations)+str(n_epochs), 'a+') as f:
    f.write('-------------Original-------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs))
    f.write('All Prec.s:{}\n'.format(prec_all))
    f.write('All Rec.s:{}\n'.format(rec_all))
    f.write('All F1.s:{}\n'.format(f1_all))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))
    f.write('Average F1 by class (Real / Fake): {}, {} \n'.format(sum(f1_r_all) / iterations, sum(f1_f_all) / iterations))

    f.write('\n-------------RestyleA------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs_res))
    f.write('All Prec.s:{}\n'.format(prec_all_res))
    f.write('All Rec.s:{}\n'.format(rec_all_res))
    f.write('All F1.s:{}\n'.format(f1_all_res))    
    f.write('Average acc.: {} \n'.format(sum(test_accs_res) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_res) /iterations, sum(rec_all_res) /iterations, sum(f1_all_res) / iterations))
    f.write('Average F1 by class (Real / Fake): {}, {} \n'.format(sum(f1_r_all_res) / iterations, sum(f1_f_all_res) / iterations))
    
    f.write('\n-------------RestyleB------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs_resb))
    f.write('All Prec.s:{}\n'.format(prec_all_resb))
    f.write('All Rec.s:{}\n'.format(rec_all_resb))
    f.write('All F1.s:{}\n'.format(f1_all_resb))    
    f.write('Average acc.: {} \n'.format(sum(test_accs_resb) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_resb) /iterations, sum(rec_all_resb) /iterations, sum(f1_all_resb) / iterations))
    f.write('\n-------------RestyleC------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs_resc))
    f.write('All Prec.s:{}\n'.format(prec_all_resc))
    f.write('All Rec.s:{}\n'.format(rec_all_resc))      
    f.write('All F1.s:{}\n'.format(f1_all_resc))    
    f.write('Average acc.: {} \n'.format(sum(test_accs_resc) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_resc) /iterations, sum(rec_all_resc) /iterations, sum(f1_all_resc) / iterations))
    f.write('\n-------------RestyleD------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs_resd))
    f.write('All Prec.s:{}\n'.format(prec_all_resd))
    f.write('All Rec.s:{}\n'.format(rec_all_resd))      
    f.write('All F1.s:{}\n'.format(f1_all_resd))    
    f.write('Average acc.: {} \n'.format(sum(test_accs_resd) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_resd) /iterations, sum(rec_all_resd) /iterations, sum(f1_all_resd) / iterations))
