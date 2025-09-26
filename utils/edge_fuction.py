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

        edge_attr = source_node_emb + target_node_emb  # Element-wise product
        return self.mlp(edge_attr)