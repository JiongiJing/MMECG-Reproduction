import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mmecg_model import MMECGTransformer, mu_law_encode, mu_law_decode

class MMECGDataset(Dataset):
    """
    Dataset class for loading processed MMECG data
    """
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing processed .npz files
        """
        self.data_dir = data_dir
        self.chunk_files = [f for f in os.listdir(data_dir) if f.startswith('data_chunk_') and f.endswith('.npz')]
        self.chunk_files.sort()
        
        # Count total samples
        self.total_samples = 0
        self.chunk_info = []
        
        for chunk_file in self.chunk_files:
            data = np.load(os.path.join(data_dir, chunk_file))
            num_samples = len(data['rcg'])
            self.chunk_info.append({
                'file': chunk_file,
                'num_samples': num_samples,
                'start_idx': self.total_samples
            })
            self.total_samples += num_samples
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which chunk contains this index
        for chunk in self.chunk_info:
            if idx < chunk['start_idx'] + chunk['num_samples']:
                chunk_idx = idx - chunk['start_idx']
                
                # Load data from chunk
                data = np.load(os.path.join(self.data_dir, chunk['file']))
                
                rcg = data['rcg'][chunk_idx]  # [50, 1, seq_len]
                ecg = data['ecg'][chunk_idx]  # [seq_len]
                positions = data['positions'][chunk_idx]  # [50, 3]
                
                # Apply μ-law encoding to ECG
                ecg_normalized = 2 * (ecg - ecg.min()) / (ecg.max() - ecg.min()) - 1
                ecg_mu_law = mu_law_encode(torch.FloatTensor(ecg_normalized), mu=255)
                ecg_quantized = ((ecg_mu_law + 1) * 127.5).long()
                ecg_quantized = torch.clamp(ecg_quantized, 0, 255)
                
                # Convert to torch tensors
                rcg_tensor = torch.FloatTensor(rcg)
                positions_tensor = torch.FloatTensor(positions)
                
                return {
                    'radar': rcg_tensor,
                    'ecg': ecg_quantized,
                    'positions': positions_tensor
                }
        
        raise IndexError(f"Index {idx} out of range")

class MMECGTrainer:
    """
    Training class for MMECG model following paper specifications
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Training hyperparameters from paper
        self.lr = 0.002
        self.batch_size = 64
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Dynamic learning rate scheduling
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-5)
        self.warmup_epochs = 5
        
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
        
        # Add progress bar
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
        progress_bar.close()
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Add progress bar for validation
        from tqdm import tqdm
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                radar = batch['radar'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                positions = batch['positions'].to(self.device)
                
                logits = self.model(radar, positions)
                loss = self.criterion(logits.reshape(-1, 256), ecg.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
                
        progress_bar.close()
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """Complete training loop"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Add overall progress bar for epochs
        from tqdm import tqdm
        epoch_bar = tqdm(range(num_epochs), desc="Epochs", position=0)
        
        best_val_loss = float('inf')
        
        for epoch in epoch_bar:
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate warmup and scheduling
            if epoch < self.warmup_epochs:
                # Linear warmup
                lr_scale = min(1.0, float(epoch + 1) / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * lr_scale
            else:
                self.scheduler.step()
            
            # Update epoch progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val': f'{best_val_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Detailed epoch logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_mmecg_model.pth')
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  Saved best model at epoch {epoch+1} (val_loss: {val_loss:.4f})")
        
        epoch_bar.close()
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        # Set matplotlib backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MMECG Training Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png')
        plt.close()  # Close figure to free memory

    def visualize_reconstruction(self, val_loader, num_samples=5):
        """
        Visualize reconstruction results for sample validation data
        Args:
            val_loader: Validation data loader
            num_samples: Number of samples to visualize
        """
        # Set matplotlib backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
        import matplotlib.pyplot as plt
        
        self.model.eval()
        
        # Get a batch of validation data
        batch = next(iter(val_loader))
        radar = batch['radar'].to(self.device)[:num_samples]
        ecg_true = batch['ecg'].to(self.device)[:num_samples]
        positions = batch['positions'].to(self.device)[:num_samples]
        
        with torch.no_grad():
            # Get model predictions
            logits = self.model(radar, positions)
            
            # Convert logits to predicted ECG values
            pred_probs = torch.softmax(logits, dim=-1)
            ecg_pred_indices = torch.argmax(pred_probs, dim=-1)
            
            # Convert quantized indices back to continuous values
            # ecg_pred_quantized = (ecg_pred_indices.float() / 127.5) - 1
            ecg_pred = mu_law_decode(ecg_pred_indices, mu=255)
            
            # Convert true quantized ECG back to continuous
            ecg_true_continuous = mu_law_decode(ecg_true, mu=255)
        
        # Calculate metrics
        import numpy as np
        
        correlations = []
        rmses = []
        
        for i in range(num_samples):
            true_sig = ecg_true_continuous[i].cpu().numpy()
            pred_sig = ecg_pred[i].cpu().numpy()
            
            # Calculate Pearson correlation
            corr, _ = pearsonr(true_sig, pred_sig)
            correlations.append(corr)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((true_sig - pred_sig) ** 2))
            rmses.append(rmse)
        
        # Create reconstruction visualization
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        time = np.linspace(0, 3.2, 640)  # 3.2 seconds at 200Hz
        
        for i, ax in enumerate(axes):
            true_sig = ecg_true_continuous[i].cpu().numpy()
            pred_sig = ecg_pred[i].cpu().numpy()
            
            ax.plot(time, true_sig, 'b-', label='Ground Truth', alpha=0.7, linewidth=1.5)
            ax.plot(time, pred_sig, 'r-', label='Reconstructed', alpha=0.7, linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ECG Amplitude')
            ax.set_title(f'Sample {i+1}: Correlation={correlations[i]:.3f}, RMSE={rmses[i]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reconstruction_results.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        # Print summary statistics
        print(f"\nReconstruction Quality Summary:")
        print(f"Average Pearson Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
        print(f"Average RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"Best Correlation: {np.max(correlations):.4f}")
        print(f"Worst Correlation: {np.min(correlations):.4f}")

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
    
    # Load real processed data
    data_dir = r"C:\Users\28474\Desktop\dataset\MMECG\split"
    print(f"Loading real data from {data_dir}...")
    
    # Create dataset from processed data
    full_dataset = MMECGDataset(data_dir)
    
    # Split into train and validation (80/20) with fixed random seed for reproducibility
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    # Set random seed for reproducible splitting
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Create data loaders with smaller batch size for memory constraints
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)
    
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: 32")
    
    # Initialize model and trainer
    model = MMECGTransformer()
    trainer = MMECGTrainer(model)
    
    # Start training
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Visualize reconstruction results
    print("\nVisualizing reconstruction results...")
    trainer.visualize_reconstruction(val_loader, num_samples=5)
    
    print("Training completed!")

if __name__ == "__main__":
    main()