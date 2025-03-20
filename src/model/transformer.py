import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Split into multiple heads
        query = self.query(query).reshape(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(key).reshape(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(value).reshape(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        
        # Compute context vector
        out = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_size)
        out = self.fc_out(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        
        # Feed-forward with residual connection and layer normalization
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        out = self.dropout(out)
        
        return out


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=512,
        num_layers=6,
        heads=8,
        forward_expansion=4,
        dropout=0.1,
        max_length=100,
        num_classes=2,
    ):
        super(SimpleTransformer, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                ) for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_size, num_classes)
    
    def forward(self, x, mask=None):
        batch_size, seq_length = x.shape
        
        # Create position indices
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(x.device)
        
        # Get embeddings and add positional encoding
        out = self.word_embedding(x) + self.position_embedding(positions)
        out = self.dropout(out)
        
        # Pass through transformer blocks
        for layer in self.layers:
            out = layer(out, mask)
        
        # Global average pooling for classification
        out = torch.mean(out, dim=1)
        out = self.classifier(out)
        
        return out


def create_bert_like_model(vocab_size, hidden_size=768, num_layers=12, num_heads=12, num_classes=2):
    """
    Create a BERT-like transformer model.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden size of the transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_classes: Number of output classes
        
    Returns:
        A BERT-like transformer model
    """
    return SimpleTransformer(
        vocab_size=vocab_size,
        embed_size=hidden_size,
        num_layers=num_layers,
        heads=num_heads,
        forward_expansion=4,
        dropout=0.1,
        max_length=512,
        num_classes=num_classes,
    ) 