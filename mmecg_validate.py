import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import json

from mmecg_model import MMECGTransformer, mu_law_encode, mu_law_decode

class MMECGValidationDataset:
    """
    Dataset class for loading processed MMECG validation data
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
                
                # Apply μ-law encoding to ECG (same as training)
                ecg_normalized = 2 * (ecg - ecg.min()) / (ecg.max() - ecg.min()) - 1
                ecg_mu_law = mu_law_encode(torch.FloatTensor(ecg_normalized), mu=255)
                ecg_quantized = ((ecg_mu_law + 1) * 127.5).long()
                ecg_quantized = torch.clamp(ecg_quantized, 0, 255)
                
                # Convert to torch tensors
                rcg_tensor = torch.FloatTensor(rcg)
                positions_tensor = torch.FloatTensor(positions)
                
                return {
                    'rcg': rcg_tensor,
                    'ecg': ecg_quantized,
                    'positions': positions_tensor
                }
        
        raise IndexError(f"Index {idx} out of range")

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load trained MMECG model from checkpoint
    """
    model = MMECGTransformer()
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}, epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model file {model_path} not found!")
        return None
    
    model = model.to(device)
    model.eval()
    return model

def calculate_metrics(true_ecg, pred_ecg):
    """
    Calculate validation metrics between true and predicted ECG signals
    """
    metrics = {}
    
    # Pearson correlation
    corr, _ = pearsonr(true_ecg, pred_ecg)
    metrics['correlation'] = corr
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(true_ecg, pred_ecg))
    metrics['rmse'] = rmse
    
    # MAE
    mae = np.mean(np.abs(true_ecg - pred_ecg))
    metrics['mae'] = mae
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(true_ecg ** 2)
    noise_power = np.mean((true_ecg - pred_ecg) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    metrics['snr'] = snr
    
    return metrics

def validate_model(model, val_loader, device, output_dir='validation_results'):
    """
    Validate the model on validation dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    all_metrics = []
    all_true_ecg = []
    all_pred_ecg = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            rcg = batch['rcg'].to(device)
            true_ecg = batch['ecg'].to(device)
            positions = batch['positions'].to(device)
            
            # Get model predictions
            pred_ecg = model.predict(rcg, positions)
            
            # Convert to numpy for metric calculation
            true_ecg_np = true_ecg.cpu().numpy()
            pred_ecg_np = pred_ecg.cpu().numpy()
            
            # Calculate metrics for each sample in batch
            for i in range(len(true_ecg_np)):
                metrics = calculate_metrics(true_ecg_np[i], pred_ecg_np[i])
                all_metrics.append(metrics)
                all_true_ecg.append(true_ecg_np[i])
                all_pred_ecg.append(pred_ecg_np[i])
    
    return all_metrics, all_true_ecg, all_pred_ecg

def visualize_reconstruction(true_ecg, pred_ecg, metrics, save_path, num_samples=4):
    """
    Visualize ECG reconstruction results - max 4 subplots with random sample selection
    """
    # Set matplotlib backend to avoid display issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
    
    # Randomly select samples to visualize (max 4)
    n_samples = min(num_samples, len(true_ecg))
    if n_samples > 0:
        # Generate random indices for sample selection
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        selected_indices = rng.choice(len(true_ecg), size=n_samples, replace=False)
        
        # Create figure with appropriate size based on number of subplots
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = [axes]  # Make it iterable
        
        for idx, ax in zip(selected_indices, axes):
            time = np.linspace(0, 3.2, len(true_ecg[idx]))  # 3.2 seconds at 200Hz
            
            ax.plot(time, true_ecg[idx], 'b-', label='Ground Truth', alpha=0.8, linewidth=1.5)
            ax.plot(time, pred_ecg[idx], 'r-', label='Reconstructed', alpha=0.8, linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ECG Amplitude')
            ax.set_title(f'Sample {idx+1}: Corr={metrics[idx]["correlation"]:.3f}, RMSE={metrics[idx]["rmse"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualized {n_samples} randomly selected samples: indices {selected_indices + 1}")
    else:
        print("No samples available for visualization")

def save_results(metrics, true_ecg, pred_ecg, output_dir):
    """
    Save validation results to files
    """
    # Save metrics
    metrics_summary = {
        'average_correlation': np.mean([m['correlation'] for m in metrics]),
        'average_rmse': np.mean([m['rmse'] for m in metrics]),
        'average_mae': np.mean([m['mae'] for m in metrics]),
        'average_snr': np.mean([m['snr'] for m in metrics]),
        'median_correlation': np.median([m['correlation'] for m in metrics]),
        'median_rmse': np.median([m['rmse'] for m in metrics]),
        'all_metrics': metrics
    }
    
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Save sample predictions
    np.savez_compressed(
        os.path.join(output_dir, 'validation_predictions.npz'),
        true_ecg=np.array(true_ecg),
        pred_ecg=np.array(pred_ecg)
    )
    
    return metrics_summary

def validate_model_on_dataset():
    """
    Main validation function with parameters defined internally
    """
    # Define parameters directly in the function
    model_path = 'best_mmecg_model.pth'  # Path to trained model
    data_dir = r"C:\Users\28474\Desktop\dataset\MMECG\split"  # Processed data directory
    batch_size = 32  # Batch size for validation
    output_dir = 'validation_results'  # Output directory
    num_samples = 5  # Number of samples to visualize (max 5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device
    val_split_ratio = 0.2  # Validation split ratio (same as training)
    random_seed = 42  # Random seed for reproducible splitting (same as training)
    
    print("=== MMECG Model Validation ===")
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    print(f"Validation split ratio: {val_split_ratio}")
    print(f"Random seed: {random_seed}")
    
    # Load model
    model = load_model(model_path, device)
    if model is None:
        return
    
    # Create full dataset and split using same logic as training
    full_dataset = MMECGValidationDataset(data_dir)
    total_samples = len(full_dataset)
    val_size = int(val_split_ratio * total_samples)
    train_size = total_samples - val_size
    
    # Use same random seed as training for reproducible splitting
    generator = torch.Generator().manual_seed(random_seed)
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    print(f"Total samples: {total_samples}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training samples (excluded): {train_size}")
    
    # Perform validation
    all_metrics, all_true_ecg, all_pred_ecg = validate_model(model, val_loader, device, output_dir)
    
    # Save results
    metrics_summary = save_results(all_metrics, all_true_ecg, all_pred_ecg, output_dir)
    
    # Visualize results
    visualize_reconstruction(all_true_ecg, all_pred_ecg, all_metrics, 
                           os.path.join(output_dir, 'reconstruction_plot.png'), 
                           num_samples)
    
    # Print summary
    print("\n=== Validation Results Summary ===")
    print(f"Average Pearson Correlation: {metrics_summary['average_correlation']:.4f}")
    print(f"Average RMSE: {metrics_summary['average_rmse']:.4f}")
    print(f"Average MAE: {metrics_summary['average_mae']:.4f}")
    print(f"Average SNR: {metrics_summary['average_snr']:.2f} dB")
    print(f"Median Correlation: {metrics_summary['median_correlation']:.4f}")
    print(f"Median RMSE: {metrics_summary['median_rmse']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return metrics_summary

if __name__ == "__main__":
    validate_model_on_dataset()