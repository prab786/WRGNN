class EdgeEmb(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeEmb, self).__init__()
        # Just the linear transformations
        self.v1 = Linear(in_channels, in_channels)  # source transformation
        self.v2 = Linear(in_channels, in_channels)  # target transformation
        
        # MLP for edge embedding with GeGLU activation
        self.mlp = Seq(
            Linear(2 * in_channels, out_channels * 2),  # Double the size for GeGLU
            GeGLU(),
            LayerNorm(out_channels)  # After GeGLU reduces back to out_channels
        )
    
    def forward(self, node_embeddings, edge_index):
        # Direct transformations without normalization
        source_node_emb = self.v1(node_embeddings[edge_index[0]])
        target_node_emb = self.v2(node_embeddings[edge_index[1]])
        
        # Concatenate and process
        edge_attr = torch.cat([source_node_emb, target_node_emb], dim=-1)
        return self.mlp(edge_attr)
    
class BilinearEdgeEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearEdgeEmb, self).__init__()
        self.bilinear = nn.Bilinear(in_channels, in_channels, out_channels)

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]
        target_node_emb = node_embeddings[edge_index[1]]

        edge_attr = self.bilinear(source_node_emb, target_node_emb)
        return edge_attr

from torch.nn import Sequential as Seq, Linear, ReLU, Softmax

class ImprovedAttentionEdgeEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedAttentionEdgeEmb, self).__init__()
        self.attn_weights = nn.Parameter(torch.randn(in_channels))  # Learnable attention weights
        self.softmax = Softmax(dim=-1)  # Normalize attention
        self.mlp = Seq(
            Linear(in_channels, out_channels),
            ReLU()
        )

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]
        target_node_emb = node_embeddings[edge_index[1]]

        attn_source = source_node_emb * self.softmax(self.attn_weights)
        attn_target = target_node_emb * self.softmax(self.attn_weights)
        edge_attr = attn_source + attn_target  # Sum operation

        return self.mlp(edge_attr)
class DifferenceEdgeEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DifferenceEdgeEmb, self).__init__()
        self.mlp = Seq(
            Linear(in_channels, out_channels),
            ReLU()
        )

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]
        target_node_emb = node_embeddings[edge_index[1]]

        edge_attr = source_node_emb - target_node_emb  # Difference operation
        return self.mlp(edge_attr)
class AttentionEdgeEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attn = nn.Linear(2 * in_channels, 1)

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]
        target_node_emb = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([source_node_emb, target_node_emb], dim=-1)
        attn_weights = F.softmax(self.attn(edge_emb), dim=0)
        return attn_weights * edge_emb


class MultiHeadAttentionEdgeEmb1(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(MultiHeadAttentionEdgeEmb1, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # Head size per attention head
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        # Learnable attention weight projections for source & target nodes
        self.attn_source = nn.Linear(in_channels, in_channels, bias=False)
        self.attn_target = nn.Linear(in_channels, in_channels, bias=False)

        # Linear transformation after concatenating multi-head outputs
        self.final_proj = Linear(in_channels*2, out_channels)

        self.mlp = Seq(
            ReLU(),
            Linear(out_channels, out_channels)
        )

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]  # (num_edges, in_channels)
        target_node_emb = node_embeddings[edge_index[1]]  # (num_edges, in_channels)

        # Compute attention scores separately for source and target nodes
        attn_scores_source = torch.sigmoid(self.attn_source(source_node_emb))
        attn_scores_target = torch.sigmoid(self.attn_target(target_node_emb))

        # Apply attention scores
        attn_source = attn_scores_source * source_node_emb
        attn_target = attn_scores_target * target_node_emb

        # Concatenate multi-head outputs
        edge_attr = torch.cat([attn_source, attn_target], dim=-1)

        # Apply final projection and MLP
        edge_attr = self.final_proj(edge_attr)
        return self.mlp(edge_attr)

class MultiHeadAttentionEdgeEmb(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.in_channels=in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # Split feature space per head
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        # Attention projections (from full feature space to per-head space)
        self.attn_source = nn.Linear(in_channels, num_heads, bias=False)
        self.attn_target = nn.Linear(in_channels, num_heads, bias=False)

        # Final projection after combining heads
        self.final_proj = nn.Linear(in_channels, out_channels)

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]  # (num_edges, in_channels)
        target_node_emb = node_embeddings[edge_index[1]]  # (num_edges, in_channels)

        # Compute attention scores (num_edges, num_heads)
        attn_source = torch.sigmoid(self.attn_source(source_node_emb))
        attn_target = torch.sigmoid(self.attn_target(target_node_emb))

        # Reshape embeddings for multiple heads (num_edges, num_heads, head_dim)
        source_node_emb = source_node_emb.view(-1, self.num_heads, self.head_dim)
        target_node_emb = target_node_emb.view(-1, self.num_heads, self.head_dim)

        # Apply attention per head
        attn_source = attn_source.unsqueeze(-1) * source_node_emb
        attn_target = attn_target.unsqueeze(-1) * target_node_emb

        # Aggregate across heads (Sum or Mean)
        edge_attr = (attn_source + attn_target).view(-1, self.in_channels)  # Flatten

        return self.final_proj(edge_attr)
    
class HadamardEdgeEmb(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HadamardEdgeEmb, self).__init__()
        self.mlp = Seq(
            Linear(in_channels, out_channels),  # Only one in_channels because Hadamard reduces dimensionality
            ReLU()
        )

    def forward(self, node_embeddings, edge_index):
        source_node_emb = node_embeddings[edge_index[0]]
        target_node_emb = node_embeddings[edge_index[1]]

        edge_attr = source_node_emb * target_node_emb  # Element-wise product
        return self.mlp(edge_attr)
       
class MaskAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)
    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores

# class BertClassifier(nn.Module):
#     def __init__(self, n_classes):
#         super(BertClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(p = 0.5)
#         #self.gat1 = GATConv(self.bert.config.hidden_size, self.bert.config.hidden_size, heads=1, concat=True)
#         self.ee=ImprovedAttentionEdgeEmb(self.bert.config.hidden_size,self.bert.config.hidden_size)
#         #self.ee1=HadamardEdgeEmb(self.bert.config.hidden_size,self.bert.config.hidden_size)        
#         self.gate_nn = torch.nn.Sequential(
#         torch.nn.Linear(self.bert.config.hidden_size, 1),  # Input feature dimension 
#         torch.nn.Sigmoid()      # Apply sigmoid to get attention weights
#         )

#         # Initialize GlobalAttention layer
#         self.attention_pool = GlobalAttention(self.gate_nn)
#         self.attention=MaskAttention(self.bert.config.hidden_size)
#         self.fc_out = nn.Linear(self.bert.config.hidden_size, n_classes)
#         self.binary_transform = nn.Linear(self.bert.config.hidden_size, 2)
#         self.fc = nn.Linear(self.bert.config.hidden_size, 2)
#         self.act=torch.nn.LeakyReLU()
#         self.all_edge_attentions = []  
#         self.all_edge_indices = []  
#         self.all_edge_batches = [] 

#     def forward(self, input_ids, attention_mask,graph_data):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_outputs = outputs[1]
#         pooled_outputs = self.dropout(pooled_outputs)
        
#         #binary_output = self.binary_transform(pooled_outputs)
#         x1=outputs['last_hidden_state']
#         x= x1[:, 1:, :]
#         x=x.flatten(start_dim=0, end_dim=1)
        
#         edge_index = graph_data.edge_index
#         # #print(graph_data)
#         edge_batch = graph_data.batch[edge_index[0]]   
#         # #x = self.gat1(x, edge_index) 
#         # #  
#         EdgeEmbeding = self.ee(x, edge_index) 
#         edge_attention = self.gate_nn(EdgeEmbeding).squeeze(-1) 
#         self.all_edge_attentions.append(edge_attention.detach().cpu())
#         self.all_edge_indices.append(edge_index.detach().cpu())  # Save edge (source, target)
#         self.all_edge_batches.append(edge_batch.detach().cpu())  # Save corresponding graph index
#         #atten_edge=self.gate_nn(EdgeEmbeding, edge_batch)
#         graph_embeddings = self.attention_pool(EdgeEmbeding, edge_batch)
#         #graph_embeddings = self.dropout(graph_embeddings)
#         #graph_embeddings = self.dropout(graph_embeddings)
#         #rep=torch.cat((graph_embeddings, pooled_outputs), dim=1)
#         rep = torch.stack((graph_embeddings, pooled_outputs), dim=1)
#         rep, scores = self.attention(rep)
#         output = self.fc(rep)
#          #+pooled_outputs
#         #rep=self.act(rep)
#         #output2=self.binary_transform(graph_embeddings)
#         output3 = self.fc_out(rep)
#         return   output3,output
# from torch_scatter import scatter
class GeGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)  
class MaxPoolGNNLayer(MessagePassing):
    def __init__(self,in_channels,out_channels):
        super(MaxPoolGNNLayer, self).__init__(aggr='sum')
        self.mlp = Seq(
            Linear(in_channels, out_channels),  # Only one in_channels because Hadamard reduces dimensionality
            ReLU()
        ) 
    def forward(self, x, edge_index, edge_weight):
        # x: Node features [num_nodes, feature_dim] (say, 786)
        # edge_index: [2, num_edges]
        # edge_weight: [num_edges, 786] -> embedding per edge
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # x_j: [num_edges, 786] — source node features per edge
        # edge_weight: [num_edges, 786] — edge features
        return  edge_weight  # Element-wise multiplication

    def update(self, aggr_out):
        # aggr_out: [num_nodes, 786]
        return self.mlp(aggr_out)

class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        self.dropout = nn.Dropout(p=0.1)
        self.ee = EdgeEmb(self.bert.config.hidden_size, self.bert.config.hidden_size)
        #self.ee2 = DifferenceEdgeEmb(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.node_model=MaxPoolGNNLayer(self.bert.config.hidden_size,self.bert.config.hidden_size)
        self.gate_nn=nn.Sequential(
                    nn.Linear(self.bert.config.hidden_size, 128),
                    nn.GELU(),
                    nn.Linear(128, 1,bias=True)
                )
        # for m in self.gate_nn.modules():
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.xavier_uniform_(m.weight)

        #self.attention_pool = GlobalAttention(self.gate_nn)
        self.graph_pool = AttentionalAggregation(
            self.gate_nn
        )

        self.graph_pool1 = AttentionalAggregation(
            gate_nn=nn.Sequential(
                    nn.Linear(self.bert.config.hidden_size, 128),
                    nn.GELU(),
                    nn.Linear(128, 1,bias=True)
                )
        )
        self.attention = MaskAttention(self.bert.config.hidden_size)
        self.fc_out = nn.Linear(self.bert.config.hidden_size, 2)
        #self.binary_transform = nn.Linear(self.roberta.config.hidden_size*2, 2)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        self.fc2 = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.fc_out2 = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.act = torch.nn.LeakyReLU()

        # Storage for edge attentions in test mode only
        self.test_edge_attentions = None  

    def forward(self, input_ids, attention_mask, graph_data, store_attention=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)
        all_hidden_states = outputs.hidden_states


        x1 = all_hidden_states[2]
        x = x1[:, 1:, :]
        x = x.flatten(start_dim=0, end_dim=1)

        edge_index = graph_data.edge_index
        edge_type=graph_data.edge_type
        edge_batch = graph_data.batch[edge_index[0]]  
        
        EdgeEmbedding = self.ee(x, edge_index) 
       # EdgeEmbedding2 = self.ee2(x, edge_index) 
        
        x_updated = self.node_model(x, edge_index, EdgeEmbedding)
        # print(x_updated.shape)
        EdgeEmbedding2 = self.ee(x_updated, edge_index)
        # Compute edge attention
        edge_attention = self.gate_nn(EdgeEmbedding).squeeze(-1)  # Shape: [num_edges]

        # Store edge attention only in test mode
        if store_attention:
            # print(input_ids.shape)
            # print(graph_data)
            self.test_edge_attentions = (edge_index.detach().cpu(), edge_attention.detach().cpu(), edge_batch.detach().cpu(),edge_type.detach().cpu())

        # Apply attention pooling
        graph_embeddings = self.graph_pool(EdgeEmbedding2, edge_batch)
        # graph_embeddings2 = self.graph_pool1(EdgeEmbedding2, edge_batch)
        #graph_embeddings = self.dropout(graph_embeddings)
        # graph_embeddings2 = self.dropout(graph_embeddings2)
        # rep1 = torch.stack((graph_embeddings, graph_embeddings2), dim=1)
        # rep1, scores = self.attention(rep1)
        #rep=torch.cat((rep1, pooled_outputs), dim=1)
        
        output = self.fc(graph_embeddings)
        output_g = self.fc2(graph_embeddings)
         #+pooled_outputs
        #rep=self.act(rep)
        #output2=self.binary_transform(graph_embeddings)
        output3 = self.fc_out(pooled_outputs)
        output3_p = self.fc_out2(pooled_outputs)
        return  output3_p, output3,output_g,output

        return output3, output
    
    def get_edge_attention(self, save_to_file=False, filename="edge_attentions.csv"):
        """Returns stored edge attentions from test mode and optionally saves to a CSV file."""
        if self.test_edge_attentions is None:
            raise ValueError("Edge attentions have not been stored. Run the model with store_attention=True in test mode.")

        edge_index, edge_attention, edge_batch,edge_type = self.test_edge_attentions

        # Convert to DataFrame
        df = pd.DataFrame({
            "graph_id": edge_batch.tolist(),
            "source_node": edge_index[0].tolist(),
            "target_node": edge_index[1].tolist(),
            "edge_type":edge_type.tolist(),
            "attention_score": edge_attention.tolist()
        })

        if save_to_file:
            df.to_csv(filename, index=False)
            #print(f"Edge attentions saved to {filename}")

        return df

def create_train_loader(contents, contents_aug1, contents_aug2, labels, fg_label, aug_fg1, aug_fg2, tokenizer, max_len, batch_size,graphs,x_train_res1_graph,x_train_res2_graph):
    ds = NewsDatasetAug(texts = contents, aug_texts1 = contents_aug1, aug_texts2 = contents_aug2, labels = np.array(labels), \
                        fg_label = fg_label, aug_fg1 = aug_fg1, aug_fg2 = aug_fg2, tokenizer=tokenizer, max_len=max_len,graphs=graphs,x_train_res1_graph=x_train_res1_graph,x_train_res2_graph=x_train_res2_graph)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=5)

def create_eval_loader(contents, labels, tokenizer, max_len, batch_size,graphs):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer=tokenizer, max_len=max_len, graphs=graphs )
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)
def create_eval_loader2(contents, labels, tokenizer, max_len, batch_size,graphs):
    ds = NewsDataset2(texts = contents, labels = np.array(labels), tokenizer=tokenizer, max_len=max_len, graphs=graphs )
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

def create_graph_loader(contents, labels, tokenizer, max_len, batch_size):
    ds = NewsDataset(texts = contents, labels = np.array(labels), tokenizer=tokenizer, max_len=max_len)
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)
