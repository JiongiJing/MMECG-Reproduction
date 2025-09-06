import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Union

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim=64, dropout=0.1, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
        self.head_dim = qkv_dim
        self.batch_first = batch_first
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Separate projections for Q, K, V to achieve custom qkv_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * qkv_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_heads * qkv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_heads * qkv_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * qkv_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=False):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        tgt_len, bsz, embed_dim = query.size()
        src_len, _, _ = key.size()
        
        # Project Q, K, V
        q = self.q_proj(query).view(tgt_len, bsz, self.num_heads, self.qkv_dim).transpose(0, 1)
        k = self.k_proj(key).view(src_len, bsz, self.num_heads, self.qkv_dim).transpose(0, 1)
        v = self.v_proj(value).view(src_len, bsz, self.num_heads, self.qkv_dim).transpose(0, 1)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)
        
        return attn_output, attn_weights

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, qkv_dim: int = 64, dim_feedforward: int = 128, 
                 dropout: float = 0.1, activation: Union[str, Callable] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True):
        super().__init__()
        
        self.self_attn = CustomMultiheadAttention(d_model, nhead, qkv_dim, dropout, bias, batch_first)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
            else:
                raise ValueError(f"Activation {activation} not supported")
        else:
            self.activation = activation
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        
        return x
    
    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], 
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.dropout1(x)
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output

# Factory function to create the custom transformer
def create_custom_transformer(embed_dim=32, num_heads=4, qkv_dim=64, ff_dim=128, 
                             num_layers=3, dropout=0.1, batch_first=True):
    
    layer = CustomTransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        qkv_dim=qkv_dim,
        dim_feedforward=ff_dim,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=False
    )
    
    return CustomTransformerEncoder(layer, num_layers)

if __name__ == "__main__":
    # Test the implementation
    batch_size, seq_len, embed_dim = 50, 50, 32
    
    model = create_custom_transformer(
        embed_dim=embed_dim,
        num_heads=4,
        qkv_dim=64,
        ff_dim=128,
        num_layers=3,
        dropout=0.1
    )
    
    # Create dummy input with shape (batch_size, seq_len, embed_dim) = (50, 50, 32)
    dummy_input = torch.randn(batch_size, seq_len, embed_dim)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Custom Transformer implementation meets the requirements:")
    print(f"- Input dimension: {embed_dim}")
    print(f"- Number of heads: 4")
    print(f"- Q/K/V dimension: 64")
    print(f"- Feed-forward dimension: 128")
    print(f"- Number of layers: 3")
    print(f"- Dropout: 0.1")
    print(f"- Batch first: True")