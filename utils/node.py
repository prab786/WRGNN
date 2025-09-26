import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm,Dropout
import warnings
from torch.nn import Sequential as Seq, Linear, ReLU,SiLU,GELU,Tanh,LeakyReLU,SELU
warnings.filterwarnings("ignore")

class GeGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)   
# class MaxPoolGNNLayer(MessagePassing):
#     def __init__(self,in_channels,out_channels):
#         super(MaxPoolGNNLayer, self).__init__(aggr='sum')
#         self.mlp = Seq(
#             Linear(in_channels, out_channels),  # Only one in_channels because Hadamard reduces dimensionality
#             LayerNorm(out_channels),
#             GELU()
#         ) 
#     def forward(self, x, edge_index, edge_weight):
#         # x: Node features [num_nodes, feature_dim] (say, 786)
#         # edge_index: [2, num_edges]
#         # edge_weight: [num_edges, 786] -> embedding per edge
#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_j, edge_weight):
#         # x_j: [num_edges, 786] — source node features per edge
#         # edge_weight: [num_edges, 786] — edge features
#         return  edge_weight  # Element-wise multiplication

#     def update(self, aggr_out):
#         # aggr_out: [num_nodes, 786]
#         return self.mlp(aggr_out)
    


class NodeUpdateLayer(MessagePassing):
    """Enhanced node update layer with edge-aware message passing"""
    def __init__(self, node_channels, edge_channels, out_channels, aggr='add'):
        super(NodeUpdateLayer, self).__init__(aggr=aggr)
        
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
        
        # Message generation network
        self.message_net = Seq(
            Linear(node_channels + edge_channels, out_channels),
            LayerNorm(out_channels),
            GELU(),
            Dropout(0.1),
            Linear(out_channels, out_channels)
        )
        
        # Node update network
        self.update_net = Seq(
            Linear(node_channels + out_channels, out_channels),
            LayerNorm(out_channels),
            GELU(),
            Dropout(0.1),
            Linear(out_channels, out_channels)
        )
        
        # Residual connection
        if node_channels != out_channels:
            self.residual_proj = Linear(node_channels, out_channels)
        else:
            self.residual_proj = None
            
    def forward(self, node_features, edge_index, edge_embeddings):
        """
        Update node features using edge-aware message passing
        Args:
            node_features: [num_nodes, node_channels]
            edge_index: [2, num_edges]  
            edge_embeddings: [num_edges, edge_channels]
        Returns:
            updated_nodes: [num_nodes, out_channels]
        """
        # Store original features for residual connection
        residual = node_features
        
        # Perform message passing
        out = self.propagate(edge_index, x=node_features, edge_attr=edge_embeddings)
        
        # Apply residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        return out + residual
    
    def message(self, x_j, edge_attr):
        """Create messages by combining neighbor features with edge information"""
        # x_j: neighbor node features [num_edges, node_channels]
        # edge_attr: edge embeddings [num_edges, edge_channels]
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_net(message_input)
    
    def update(self, aggr_out, x):
        """Update node features using aggregated messages"""
        # aggr_out: aggregated messages [num_nodes, out_channels]
        # x: original node features [num_nodes, node_channels]
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_net(update_input)
class NodeEmbedding(MessagePassing):
    """
    Graph Neural Network layer that computes node embeddings using edge information.
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        aggr (str): Aggregation method ('sum', 'mean', 'max', 'add')
        dropout_rate (float): Dropout rate for regularization
        use_residual (bool): Whether to use residual connections
    """
    
    def __init__(self, in_channels, out_channels, aggr='sum', dropout_rate=0.1, use_residual=True):
        super(NodeEmbedding, self).__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        
        # MLP for processing aggregated messages
        self.mlp = Seq(
            Linear(in_channels, out_channels),
            LayerNorm(out_channels),
            ReLU(),
            Dropout(dropout_rate),
            Linear(out_channels, out_channels),  # Additional layer for better expressiveness
            LayerNorm(out_channels),
            ReLU()
        )
        
        # Optional residual connection projection
        if use_residual and in_channels != out_channels:
            self.residual_proj = Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
    
    def forward(self, x, edge_index, edge_embedding):
        """
        Forward pass of the NodeEmbedding layer.
        
        Args:
            x (Tensor): Node features [num_nodes, in_channels]
            edge_index (Tensor): Edge connectivity [2, num_edges]
            edge_embedding (Tensor): Edge features [num_edges, in_channels]
            
        Returns:
            Tensor: Updated node embeddings [num_nodes, out_channels]
        """
        # Store original node features for potential residual connection
        x_orig = x
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_embedding=edge_embedding)
        
        # Apply residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                x_orig = self.residual_proj(x_orig)
            out = out + x_orig
            
        return out
    
    def message(self, x_j, edge_embedding):
        """
        Create messages from source nodes and edge features.
        
        Args:
            x_j (Tensor): Source node features [num_edges, in_channels]
            edge_embedding (Tensor): Edge features [num_edges, in_channels]
            
        Returns:
            Tensor: Messages [num_edges, in_channels]
        """
        # Combine node and edge information
        # Option 1: Element-wise multiplication (Hadamard product)
        # return x_j * edge_embedding
        
        # Option 2: Concatenation + projection (more expressive)
        # combined = torch.cat([x_j, edge_embedding], dim=-1)
        # return self.edge_mlp(combined)
        
        # Option 3: Addition (simple but effective)
        return  edge_embedding
        
        # Option 4: Weighted combination
        # return 0.5 * x_j + 0.5 * edge_embedding
    
    def update(self, aggr_out):
        """
        Update node embeddings using aggregated messages.
        
        Args:
            aggr_out (Tensor): Aggregated messages [num_nodes, in_channels]
            
        Returns:
            Tensor: Updated node embeddings [num_nodes, out_channels]
        """
        return self.mlp(aggr_out)

class EnhancedNodeEmbedding(MessagePassing):
    """
    Enhanced version with attention mechanism and more sophisticated edge-node interaction.
    """
    
    def __init__(self, in_channels, out_channels, aggr='sum', num_heads=4, dropout_rate=0.1):
        super(EnhancedNodeEmbedding, self).__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Edge-node interaction MLP
        self.edge_node_mlp = Seq(
            Linear(2 * in_channels, out_channels),  # Concatenated node + edge features
            LayerNorm(out_channels),
            ReLU(),
            Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Final MLP
        self.final_mlp = Seq(
            Linear(out_channels, out_channels),
            LayerNorm(out_channels),
            ReLU(),
            Dropout(dropout_rate),
            Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_embedding):
        """Forward pass with attention mechanism."""
        return self.propagate(edge_index, x=x, edge_embedding=edge_embedding)
    
    def message(self, x_j, edge_embedding):
        """Create messages using edge-node interaction MLP."""
        # Concatenate source node features with edge features
        combined = torch.cat([x_j, edge_embedding], dim=-1)
        return self.edge_node_mlp(combined)
    
    def update(self, aggr_out):
        """Update using attention mechanism."""
        # Apply self-attention (treating each node as a sequence of length 1)
        aggr_out_expanded = aggr_out.unsqueeze(1)  # [num_nodes, 1, out_channels]
        attn_out, _ = self.attention(aggr_out_expanded, aggr_out_expanded, aggr_out_expanded)
        attn_out = attn_out.squeeze(1)  # [num_nodes, out_channels]
        
        return self.final_mlp(attn_out)