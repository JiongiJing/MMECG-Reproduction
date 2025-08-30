import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mmecg_model import MMECGTransformer, mu_law_encode, mu_law_decode

class MMECGDataset(Dataset):
    """
    Dataset class for MMECG training as described in the paper
    """
    def __init__(self, radar_signals, ecg_signals, positions, seq_len=640):
        """
        Args:
            radar_signals: [N_samples, 50, 640] - 4D cardiac motion measurements  
            ecg_signals: [N_samples, 640] - ground truth ECG signals
            positions: [N_samples, 50, 3] - 3D spatial coordinates
        """
        self.radar_signals = torch.FloatTensor(radar_signals)
        self.ecg_signals = torch.FloatTensor(ecg_signals)
        self.positions = torch.FloatTensor(positions)
        self.seq_len = seq_len
        
        # Apply μ-law encoding to ECG signals and quantize to 256 levels
        self.ecg_encoded = self.quantize_ecg(self.ecg_signals)
        
    def quantize_ecg(self, ecg_signals):
        """
        Apply μ-law transformation and quantization as described in the paper
        """
        # Normalize ECG to [-1, 1] range
        ecg_min, ecg_max = ecg_signals.min(), ecg_signals.max()
        ecg_normalized = 2 * (ecg_signals - ecg_min) / (ecg_max - ecg_min) - 1
        
        # Apply μ-law encoding
        ecg_mu_law = mu_law_encode(ecg_normalized, mu=255)
        
        # Quantize to 256 levels [0, 255]
        ecg_quantized = ((ecg_mu_law + 1) * 127.5).long()
        ecg_quantized = torch.clamp(ecg_quantized, 0, 255)
        
        return ecg_quantized
        
    def __len__(self):
        return len(self.radar_signals)
        
    def __getitem__(self, idx):
        return {
            'radar': self.radar_signals[idx].unsqueeze(1),  # Add channel dim: [50, 1, 640]
            'ecg': self.ecg_encoded[idx],                   # [640] with values [0, 255]
            'positions': self.positions[idx]                # [50, 3]
        }

class MMECGTrainer:
    """
    Training class for MMECG model following paper specifications
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Training hyperparameters from paper
        self.lr = 0.001
        self.batch_size = 64
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Cross-entropy loss for categorical distribution (as mentioned in paper)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            radar = batch['radar'].to(self.device)      # [batch, 50, 1, 640]
            ecg = batch['ecg'].to(self.device)          # [batch, 640]
            positions = batch['positions'].to(self.device)  # [batch, 50, 3]
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(radar, positions)       # [batch, 640, 256]
            
            # Compute loss
            loss = self.criterion(logits.reshape(-1, 256), ecg.reshape(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                radar = batch['radar'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                positions = batch['positions'].to(self.device)
                
                logits = self.model(radar, positions)
                loss = self.criterion(logits.reshape(-1, 256), ecg.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """Complete training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if epoch == 0 or val_loss < min(self.val_losses[:-1]):
                torch.save(self.model.state_dict(), 'best_mmecg_model.pth')
                print(f"Saved best model at epoch {epoch+1}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MMECG Training Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png')
        plt.show()

def create_synthetic_data(num_samples=1000, seq_len=640, num_signals=50):
    """
    Create synthetic data for testing (replace with real data loading)
    """
    # Synthetic radar signals (cardiac motion measurements)
    radar_signals = np.random.randn(num_samples, num_signals, seq_len) * 0.1
    
    # Synthetic ECG signals 
    t = np.linspace(0, 3.2, seq_len)  # 3.2 seconds at 200Hz
    ecg_signals = np.zeros((num_samples, seq_len))
    
    for i in range(num_samples):
        # Generate synthetic ECG with P, QRS, T waves
        heart_rate = 60 + np.random.randn() * 10  # BPM
        freq = heart_rate / 60  # Hz
        
        # Simple synthetic ECG
        ecg = np.sin(2 * np.pi * freq * t)  # Basic rhythm
        ecg += 0.5 * np.sin(2 * np.pi * freq * 3 * t)  # Harmonics
        ecg += 0.1 * np.random.randn(seq_len)  # Noise
        ecg_signals[i] = ecg
    
    # Synthetic 3D positions (voxel coordinates)
    positions = np.random.randn(num_samples, num_signals, 3) * 0.1
    
    return radar_signals, ecg_signals, positions

def main():
    """Main training script"""
    print("=== MMECG Model Training ===")
    
    # Create synthetic data (replace with real data loading)
    print("Loading synthetic data...")
    radar_data, ecg_data, positions_data = create_synthetic_data(num_samples=1000)
    
    # Split data
    split_idx = int(0.8 * len(radar_data))
    train_radar, val_radar = radar_data[:split_idx], radar_data[split_idx:]
    train_ecg, val_ecg = ecg_data[:split_idx], ecg_data[split_idx:]
    train_pos, val_pos = positions_data[:split_idx], positions_data[split_idx:]
    
    # Create datasets
    train_dataset = MMECGDataset(train_radar, train_ecg, train_pos)
    val_dataset = MMECGDataset(val_radar, val_ecg, val_pos)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model and trainer
    model = MMECGTransformer()
    trainer = MMECGTrainer(model)
    
    # Start training
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Plot results
    trainer.plot_training_curves()
    
    print("Training completed!")

if __name__ == "__main__":
    main()