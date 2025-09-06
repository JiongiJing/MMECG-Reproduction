import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from custom_transformer_compat import create_custom_transformer

def mu_law_encode(x, mu=255):
    """
    Apply μ-law encoding to ECG signal
    Args:
        x: ECG signal normalized to [-1, 1]
        mu: μ-law parameter (default 255)
    Returns:
        encoded signal
    """
    return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, dtype=x.dtype, device=x.device))

def mu_law_decode(y, mu=255):
    """
    Apply μ-law decoding 
    Args:
        y: quantized values [0, 255]
        mu: μ-law parameter (default 255)
    Returns:
        decoded ECG signal
    """
    # Convert to float and normalize to [-1, 1]
    y = y.float() / (mu / 2) - 1.0
    return torch.sign(y) * (torch.expm1(torch.abs(y) * torch.log1p(torch.tensor(mu, dtype=y.dtype, device=y.device))) / mu)

class TemporalCNNEncoder(nn.Module):
    """
    Temporal CNN Encoder as described in the paper
    Input: 50×1×640 -> Output: 50×32×80
    """
    def __init__(self, input_channels=1, output_channels=32):
        super(TemporalCNNEncoder, self).__init__()
        
        # 4 repeated layers of two 1D convolutions with kernel size 7
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels if i == 0 else output_channels, output_channels, 
                         kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(output_channels),
                nn.Conv1d(output_channels, output_channels, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.BatchNorm1d(output_channels),
                nn.MaxPool1d(2, stride=2)
            ) for i in range(4)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, N, input_channels, seq_len] where N=50, seq_len=640
        Returns:
            [batch_size, N, output_channels, seq_len/16] -> [batch_size, 50, 32, 40]
        """
        batch_size, N, C, seq_len = x.shape
        # Reshape to process all signals together
        x = x.view(batch_size * N, C, seq_len)  # [batch*N, 1, 640]
        
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # [batch*N, 32, seq_len/(2^i)]
            
        # Reshape back
        _, out_channels, out_seq_len = x.shape
        x = x.view(batch_size, N, out_channels, out_seq_len)  # [batch, 50, 32, 40]
        
        # Upsample to get 80 time steps as specified in paper Section 4.2.3:
        # "the temporal encoder output size is 50×32×80"
        x = F.interpolate(x.view(batch_size * N, out_channels, out_seq_len), 
                         size=80, mode='linear', align_corners=False)
        x = x.view(batch_size, N, out_channels, 80)  # [batch, 50, 32, 80]
        
        return x

class SpatialTransformerEncoder(nn.Module):
    """
    Spatial Transformer Encoder as described in the paper
    3 blocks of transformer with 4 heads, Q/K/V dimension 64, feed-forward 128
    """
    def __init__(self, d_model=32, nhead=4, num_layers=3, dim_feedforward=128):
        super(SpatialTransformerEncoder, self).__init__()
        
        # Linear projection for temporal features (32 -> 32)
        self.temporal_proj = nn.Linear(d_model, d_model)
        
        # Linear projection for 3D position embedding (3 -> 32)
        self.position_proj = nn.Linear(3, d_model)
        
        # Transformer encoder layers with custom Q/K/V dimension
        self.transformer = create_custom_transformer(
            embed_dim=d_model,
            num_heads=nhead,
            qkv_dim=64,
            ff_dim=dim_feedforward,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, temporal_features, positions):
        """
        Args:
            temporal_features: [batch_size, N, 32, 80]
            positions: [batch_size, N, 3] - 3D spatial coordinates
        Returns:
            [batch_size, N, 32] - spatial features
        """
        batch_size, N, d_model, seq_len = temporal_features.shape
        
        # Average temporal features across time dimension
        temporal_avg = temporal_features.mean(dim=-1)  # [batch, N, 32]
        
        # Project temporal and positional features
        temporal_proj = self.temporal_proj(temporal_avg)  # [batch, N, 32]
        position_proj = self.position_proj(positions)     # [batch, N, 32]
        
        # Combine features
        combined_features = temporal_proj + position_proj  # [batch, N, 32]
        
        # Apply transformer
        spatial_features = self.transformer(combined_features)  # [batch, N, 32]
        
        return spatial_features

class FeatureExpansion(nn.Module):
    """
    Feature expansion module to bridge encoder and decoder
    Expands features from 32 to 4 dimensions and handles temporal expansion
    """
    def __init__(self, input_dim=32, output_dim=4, target_seq_len=640):
        super(FeatureExpansion, self).__init__()
        
        # Second CNN for temporal feature expansion (as mentioned in paper)
        # 4 layers of transposed convolutions to expand from 80 to 640
        self.temporal_expansion = nn.Sequential(
            nn.ConvTranspose1d(input_dim, input_dim//2, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(input_dim//2, input_dim//4, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(input_dim//4, input_dim//8, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(input_dim//8, output_dim, kernel_size=7, stride=1, padding=3)
        )
        
        # Linear projection for spatial features
        self.spatial_expansion = nn.Linear(input_dim, output_dim * target_seq_len)
        self.target_seq_len = target_seq_len
        self.output_dim = output_dim
        
    def forward(self, temporal_features, spatial_features):
        """
        Args:
            temporal_features: [batch, N, 32, 80]
            spatial_features: [batch, N, 32]
        Returns:
            expanded_temporal: [batch, N, 4, 640]
            expanded_spatial: [batch, N, 4, 640]
        """
        batch_size, N, d_model, seq_len = temporal_features.shape
        
        # Expand temporal features
        temp_reshaped = temporal_features.view(batch_size * N, d_model, seq_len)
        expanded_temp = self.temporal_expansion(temp_reshaped)  # [batch*N, 4, expanded_len]
        
        # Adjust to exactly 640 time steps
        if expanded_temp.size(-1) != self.target_seq_len:
            expanded_temp = F.interpolate(expanded_temp, size=self.target_seq_len, mode='linear', align_corners=False)
        
        expanded_temp = expanded_temp.view(batch_size, N, self.output_dim, self.target_seq_len)
        
        # Expand spatial features
        expanded_spatial = self.spatial_expansion(spatial_features)  # [batch, N, 4*640]
        expanded_spatial = expanded_spatial.view(batch_size, N, self.output_dim, self.target_seq_len)
        
        return expanded_temp, expanded_spatial

class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network Decoder
    9 stacks of TCN with dilation factor of 2, providing receptive field of 512
    """
    def __init__(self, input_channels=4, output_classes=256, num_blocks=9, kernel_size=3):
        super(TCNDecoder, self).__init__()
        
        # Dilated causal convolution blocks
        self.tcn_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            dilation = 2 ** i  # Exponentially increasing dilation
            padding = (kernel_size - 1) * dilation  # Causal padding
            
            block = nn.Sequential(
                nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, 
                         dilation=dilation, padding=padding),
                nn.ReLU(),
                nn.BatchNorm1d(input_channels),
                nn.Dropout(0.1)
            )
            self.tcn_blocks.append(block)
            
        # Final output layer
        self.output_conv = nn.Conv1d(input_channels, output_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, 4, 640] - fused cardiac features
        Returns:
            [batch_size, 640, 256] - ECG prediction logits
        """
        # Apply TCN blocks with residual connections
        for tcn_block in self.tcn_blocks:
            residual = x
            x = tcn_block(x)
            # Remove extra padding for causality
            if x.size(-1) > residual.size(-1):
                x = x[:, :, :residual.size(-1)]
            x = x + residual  # Residual connection
            
        # Final output
        x = self.output_conv(x)  # [batch, 256, 640]
        x = x.transpose(1, 2)    # [batch, 640, 256]
        
        return x

class MMECGTransformer(nn.Module):
    """
    Complete ECG Transformer Network as described in the paper:
    "Contactless Electrocardiogram Monitoring With Millimeter Wave Radar"
    
    Architecture:
    - Input: 4D cardiac motion measurements [batch, 50, 1, 640]
    - Encoder: Hybrid CNN-Transformer for temporal-spatial features
    - Decoder: TCN for ECG reconstruction with μ-law quantization
    - Output: ECG signals [batch, 640, 256]
    """
    def __init__(self, num_signals=50, input_channels=1, seq_len=640, output_classes=256):
        super(MMECGTransformer, self).__init__()
        
        # Cardiac temporal-spatial features encoder
        self.temporal_encoder = TemporalCNNEncoder(input_channels=input_channels, output_channels=32)
        self.spatial_encoder = SpatialTransformerEncoder(d_model=32, nhead=4, num_layers=3)
        
        # Feature expansion bridge
        self.feature_expansion = FeatureExpansion(input_dim=32, output_dim=4, target_seq_len=seq_len)
        
        # ECG reconstruction decoder
        self.tcn_decoder = TCNDecoder(input_channels=4, output_classes=output_classes, num_blocks=9)
        
    def forward(self, x, positions):
        """
        Args:
            x: [batch_size, N, 1, 640] - 4D cardiac motion measurements
            positions: [batch_size, N, 3] - 3D spatial coordinates
        Returns:
            [batch_size, 640, 256] - ECG prediction logits
        """
        # Temporal feature extraction
        temporal_features = self.temporal_encoder(x)  # [batch, 50, 32, 80]
        
        # Spatial feature extraction
        spatial_features = self.spatial_encoder(temporal_features, positions)  # [batch, 50, 32]
        
        # Feature expansion
        expanded_temporal, expanded_spatial = self.feature_expansion(
            temporal_features, spatial_features
        )  # Both: [batch, 50, 4, 640]
        
        # Feature fusion (dot product as mentioned in paper)
        fused_features = expanded_temporal * expanded_spatial  # [batch, 50, 4, 640]
        
        # Reshape to [batch, 4, 640] by averaging across spatial dimension
        batch_size, N, C, L = fused_features.shape
        # Average across the spatial dimension (N) to get final features
        fused_features = fused_features.mean(dim=1)  # [batch, 4, 640]
        
        # TCN decoding
        output = self.tcn_decoder(fused_features)  # [batch, 640, 256]
        
        return output
    
    def predict(self, x, positions):
        """
        Inference method that returns decoded ECG signal
        """
        with torch.no_grad():
            logits = self.forward(x, positions)
            pred_classes = torch.argmax(logits, dim=-1)  # [batch, 640]
            ecg_signal = mu_law_decode(pred_classes)     # [batch, 640]
            return ecg_signal

# Helper function for testing
def test_model():
    """
    Test the model with random input to verify dimensions
    """
    batch_size = 2
    num_signals = 50
    seq_len = 640
    
    # Create random input
    x = torch.randn(batch_size, num_signals, 1, seq_len)
    positions = torch.randn(batch_size, num_signals, 3)
    
    # Initialize model
    model = MMECGTransformer()
    
    # Forward pass
    output = model(x, positions)
    print(f"Input shape: {x.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    ecg_pred = model.predict(x, positions)
    print(f"Predicted ECG shape: {ecg_pred.shape}")
    
    return model, output

if __name__ == "__main__":
    model, output = test_model()