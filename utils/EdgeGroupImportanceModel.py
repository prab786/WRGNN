import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax
import numpy as np


class EdgeGroupImportanceModel(nn.Module):
    """
    Model that identifies important edge groups (neighboring edge pairs) and computes their embeddings.
    Focus is on learning which edge neighborhoods are most important for the task.
    """
    
    def __init__(self, edge_dim, hidden_dim=256, num_heads=4, top_k_groups=100):
        super(EdgeGroupImportanceModel, self).__init__()
        
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.top_k_groups = top_k_groups
        
        # Multi-head attention for edge importance
        self.edge_self_attention = nn.MultiheadAttention(
            edge_dim, 
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Edge group importance scorer
        self.group_importance_scorer = EdgeGroupImportanceScorer(
            edge_dim, 
            hidden_dim, 
            num_heads
        )
        
        # Edge group embedding generator
        self.group_embedding_generator = EdgeGroupEmbeddingGenerator(
            edge_dim, 
            hidden_dim
        )
        
        # Final edge embedding refinement
        self.edge_refinement = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, edge_dim)
        )
        
    def forward(self, edge_embeddings, original_graph, transformed_graph):
        """
        Args:
            edge_embeddings: [num_edges, edge_dim] - Initial edge embeddings
            original_graph: Original graph with edge_index
            transformed_graph: Transformed graph where edges are nodes
            
        Returns:
            refined_edge_embeddings: [num_edges, edge_dim]
            edge_group_embeddings: [num_groups, hidden_dim]
            edge_group_importance: [num_groups] - Importance scores for edge groups
            important_edge_groups: List of (edge_i, edge_j) tuples
        """
        # Step 1: Self-attention on edges to capture global context
        edge_emb_attended, _ = self.edge_self_attention(
            edge_embeddings.unsqueeze(0), 
            edge_embeddings.unsqueeze(0), 
            edge_embeddings.unsqueeze(0)
        )
        edge_emb_attended = edge_emb_attended.squeeze(0)
        
        # Step 2: Find all edge groups (neighboring edge pairs)
        edge_groups = self._find_edge_groups(transformed_graph.edge_index)
        
        # Step 3: Score edge group importance
        group_importance_scores = self.group_importance_scorer(
            edge_emb_attended, 
            edge_groups, 
            original_graph
        )
        
        # Step 4: Select top-k important groups
        top_k = min(self.top_k_groups, len(edge_groups))
        top_importance, top_indices = torch.topk(group_importance_scores, top_k)
        important_edge_groups = [edge_groups[idx] for idx in top_indices]
        
        # Step 5: Generate embeddings for important edge groups
        group_embeddings = self.group_embedding_generator(
            edge_emb_attended,
            important_edge_groups,
            top_importance
        )
        
        # Step 6: Refine edge embeddings using group information
        refined_edge_embeddings = self._refine_edges_with_groups(
            edge_emb_attended,
            group_embeddings,
            important_edge_groups
        )
        
        return (
            refined_edge_embeddings,
            group_embeddings,
            top_importance,
            important_edge_groups
        )
    
    def _find_edge_groups(self, transformed_edge_index):
        """Find all neighboring edge pairs from transformed graph"""
        edge_groups = []
        num_edges = transformed_edge_index.max().item() + 1
        
        # Create adjacency list for efficiency
        adj_list = [[] for _ in range(num_edges)]
        for i in range(transformed_edge_index.shape[1]):
            src, dst = transformed_edge_index[0, i].item(), transformed_edge_index[1, i].item()
            if src != dst:  # Skip self-loops
                adj_list[src].append(dst)
        
        # Find unique edge pairs
        seen = set()
        for i in range(num_edges):
            for j in adj_list[i]:
                if i < j and (i, j) not in seen:
                    edge_groups.append((i, j))
                    seen.add((i, j))
        
        return edge_groups
    
    def _refine_edges_with_groups(self, edge_embeddings, group_embeddings, edge_groups):
        """Refine edge embeddings using group information"""
        num_edges = edge_embeddings.shape[0]
        edge_group_context = torch.zeros(
            num_edges, 
            group_embeddings.shape[1], 
            device=edge_embeddings.device
        )
        
        # Aggregate group embeddings for each edge
        edge_counts = torch.zeros(num_edges, device=edge_embeddings.device)
        
        for group_idx, (edge_i, edge_j) in enumerate(edge_groups):
            edge_group_context[edge_i] += group_embeddings[group_idx]
            edge_group_context[edge_j] += group_embeddings[group_idx]
            edge_counts[edge_i] += 1
            edge_counts[edge_j] += 1
        
        # Average the context
        mask = edge_counts > 0
        edge_group_context[mask] /= edge_counts[mask].unsqueeze(1)
        
        # Combine with original embeddings
        combined = torch.cat([edge_embeddings, edge_group_context], dim=-1)
        refined = self.edge_refinement(combined)
        
        # Residual connection
        return refined + edge_embeddings


class EdgeGroupImportanceScorer(nn.Module):
    """Scores the importance of edge groups based on various factors"""
    
    def __init__(self, edge_dim, hidden_dim, num_heads):
        super(EdgeGroupImportanceScorer, self).__init__()
        
        # Compute compatibility between edge pairs
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(edge_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
            nn.ReLU()
        )
        
        # Compute structural importance
        self.structural_scorer = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # 4 features: degrees of connected nodes
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Combine scores
        self.score_combiner = nn.Sequential(
            nn.Linear(num_heads + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, edge_embeddings, edge_groups, original_graph):
        """
        Compute importance scores for edge groups
        """
        device = edge_embeddings.device
        num_groups = len(edge_groups)
        
        # Prepare edge pair embeddings
        edge_i_embs = torch.stack([edge_embeddings[i] for i, j in edge_groups])
        edge_j_embs = torch.stack([edge_embeddings[j] for i, j in edge_groups])
        
        # Compatibility scores (multi-head)
        edge_pair_features = torch.cat([edge_i_embs, edge_j_embs], dim=-1)
        compatibility_scores = self.compatibility_scorer(edge_pair_features)
        
        # Structural scores based on node degrees
        structural_features = self._compute_structural_features(
            edge_groups, 
            original_graph.edge_index
        ).to(device)
        structural_scores = self.structural_scorer(structural_features)
        
        # Combine all scores
        all_scores = torch.cat([compatibility_scores, structural_scores], dim=-1)
        importance_scores = self.score_combiner(all_scores).squeeze(-1)
        
        return importance_scores
    
    def _compute_structural_features(self, edge_groups, edge_index):
        """Compute structural features for edge groups"""
        # Compute node degrees
        num_nodes = edge_index.max().item() + 1
        degrees = torch.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            degrees[edge_index[0, i]] += 1
            degrees[edge_index[1, i]] += 1
        
        features = []
        for edge_i, edge_j in edge_groups:
            # Get nodes connected by these edges
            nodes_i = [edge_index[0, edge_i].item(), edge_index[1, edge_i].item()]
            nodes_j = [edge_index[0, edge_j].item(), edge_index[1, edge_j].item()]
            
            # Compute degree features
            feat = torch.tensor([
                degrees[nodes_i[0]],
                degrees[nodes_i[1]],
                degrees[nodes_j[0]],
                degrees[nodes_j[1]]
            ])
            features.append(feat)
        
        return torch.stack(features)


class EdgeGroupEmbeddingGenerator(nn.Module):
    """Generates embeddings for edge groups"""
    
    def __init__(self, edge_dim, hidden_dim):
        super(EdgeGroupEmbeddingGenerator, self).__init__()
        
        # Attention-based aggregation
        self.attention_proj = nn.Linear(edge_dim * 2, 1)
        
        # Group embedding generation
        self.group_mlp = nn.Sequential(
            nn.Linear(edge_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Importance-weighted projection
        self.importance_modulation = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, edge_embeddings, edge_groups, importance_scores):
        """
        Generate embeddings for edge groups
        """
        group_embeddings = []
        
        for idx, (edge_i, edge_j) in enumerate(edge_groups):
            # Get edge embeddings
            emb_i = edge_embeddings[edge_i]
            emb_j = edge_embeddings[edge_j]
            
            # Compute attention weights
            concat_emb = torch.cat([emb_i, emb_j], dim=-1)
            attention = torch.sigmoid(self.attention_proj(concat_emb))
            
            # Weighted combination
            weighted_emb = attention * emb_i + (1 - attention) * emb_j
            combined = torch.cat([weighted_emb, concat_emb], dim=-1)
            
            # Generate group embedding
            group_emb = self.group_mlp(concat_emb)
            
            # Modulate by importance
            importance = importance_scores[idx].unsqueeze(0)
            group_emb_with_importance = torch.cat([group_emb, importance], dim=-1)
            final_emb = self.importance_modulation(group_emb_with_importance)
            
            group_embeddings.append(final_emb)
        
        return torch.stack(group_embeddings)


class EdgeGroupAnalyzer(nn.Module):
    """
    Lightweight version focused on analyzing and ranking edge groups
    """
    
    def __init__(self, edge_dim, top_k=50):
        super(EdgeGroupAnalyzer, self).__init__()
        
        self.edge_dim = edge_dim
        self.top_k = top_k
        
        # Simple but effective importance scoring
        self.importance_head = nn.Sequential(
            nn.Linear(edge_dim * 2, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, edge_embeddings, transformed_graph):
        """
        Analyze edge groups and return top-k important ones
        """
        # Get all neighboring edge pairs
        edge_pairs = []
        importance_scores = []
        
        edge_index = transformed_graph.edge_index
        for i in range(edge_index.shape[1]):
            edge_i, edge_j = edge_index[0, i].item(), edge_index[1, i].item()
            if edge_i < edge_j:  # Avoid duplicates
                # Compute importance
                pair_emb = torch.cat([
                    edge_embeddings[edge_i], 
                    edge_embeddings[edge_j]
                ], dim=-1)
                score = self.importance_head(pair_emb)
                
                edge_pairs.append((edge_i, edge_j))
                importance_scores.append(score)
        
        if importance_scores:
            importance_scores = torch.cat(importance_scores)
            
            # Get top-k
            k = min(self.top_k, len(importance_scores))
            top_scores, top_indices = torch.topk(importance_scores.squeeze(), k)
            
            top_pairs = [edge_pairs[idx] for idx in top_indices]
            
            return top_pairs, top_scores
        else:
            return [], torch.tensor([])