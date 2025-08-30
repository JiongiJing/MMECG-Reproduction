import scipy.io as sio
import numpy as np
import os
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json

class MMECGDataProcessor:
    """
    Processor for MMECG dataset to prepare it for neural network training
    """
    
    def __init__(self, data_dir, output_dir, seq_len=640, sample_step=30):
        """
        Args:
            data_dir: Directory containing .mat files
            output_dir: Directory to save processed data
            seq_len: Length of each sequence (default 640 as in paper)
            sample_step: Step size for sliding window (default 30 as in paper)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seq_len = seq_len
        self.sample_step = sample_step
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_mat_file(self, file_path):
        """Load a single .mat file and extract data"""
        try:
            data = sio.loadmat(file_path)
            struct_data = data['data'][0, 0]  # Extract the structured data
            
            # Extract fields - handle different data types correctly
            rcg_data = struct_data['RCG']  # Radar data [35505, 50]
            ecg_data = struct_data['ECG']  # ECG data [35505, 1]
            pos_data = struct_data['posXYZ']  # Position data [50, 3]
            
            # Extract scalar fields
            subject_id = int(struct_data['id'][0, 0]) if struct_data['id'].size > 0 else 0
            gender = str(struct_data['gender'][0]) if struct_data['gender'].size > 0 else "unknown"
            age = int(struct_data['age'][0, 0]) if struct_data['age'].size > 0 else 0
            physistatus = str(struct_data['physistatus'][0]) if struct_data['physistatus'].size > 0 else "unknown"
            
            # Flatten ECG to 1D array
            ecg_data = ecg_data.flatten()
            
            return {
                'rcg': rcg_data,
                'ecg': ecg_data,
                'positions': pos_data,
                'subject_id': subject_id,
                'gender': gender,
                'age': age,
                'physistatus': physistatus
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def create_sliding_windows(self, rcg_data, ecg_data, positions):
        """
        Create sliding windows from the data
        Args:
            rcg_data: [time_steps, 50] - Radar data
            ecg_data: [time_steps] - ECG data
            positions: [50, 3] - Position data (constant for all time steps)
        Returns:
            List of window segments
        """
        total_length = len(rcg_data)
        windows = []
        
        # Create sliding windows
        for start_idx in range(0, total_length - self.seq_len + 1, self.sample_step):
            end_idx = start_idx + self.seq_len
            
            # Extract window
            rcg_window = rcg_data[start_idx:end_idx]  # [seq_len, 50]
            ecg_window = ecg_data[start_idx:end_idx]  # [seq_len]
            
            # Reshape RCG to [50, seq_len] and add channel dimension
            rcg_window = rcg_window.T  # [50, seq_len]
            rcg_window = np.expand_dims(rcg_window, axis=1)  # [50, 1, seq_len]
            
            windows.append({
                'rcg': rcg_window.astype(np.float32),
                'ecg': ecg_window.astype(np.float32),
                'positions': positions.astype(np.float32)  # [50, 3]
            })
        
        return windows
    
    def process_all_files(self):
        """Process all .mat files in the directory"""
        mat_files = [f for f in os.listdir(self.data_dir) if f.endswith('.mat')]
        mat_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort numerically
        
        all_windows = []
        metadata = []
        
        print(f"Processing {len(mat_files)} .mat files...")
        
        for mat_file in tqdm(mat_files):
            file_path = os.path.join(self.data_dir, mat_file)
            data = self.load_mat_file(file_path)
            
            if data is None:
                continue
                
            # Create sliding windows
            windows = self.create_sliding_windows(
                data['rcg'], data['ecg'], data['positions']
            )
            
            # Add metadata for each window
            for window in windows:
                window_metadata = {
                    'subject_id': int(data['subject_id']),
                    'gender': str(data['gender']),
                    'age': int(data['age']),
                    'physistatus': str(data['physistatus']),
                    'source_file': mat_file
                }
                window['metadata'] = window_metadata
                
            all_windows.extend(windows)
            
            # Also store individual file metadata
            metadata.append({
                'file': mat_file,
                'subject_id': int(data['subject_id']),
                'gender': str(data['gender']),
                'age': int(data['age']),
                'physistatus': str(data['physistatus']),
                'total_windows': len(windows),
                'original_length': len(data['rcg'])
            })
        
        return all_windows, metadata
    
    def save_processed_data(self, windows, metadata):
        """Save processed data to output directory"""
        
        # Save metadata
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save windows in chunks to avoid memory issues
        chunk_size = 1000
        num_chunks = (len(windows) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(windows))
            chunk_windows = windows[start_idx:end_idx]
            
            # Prepare arrays
            rcg_chunk = np.stack([w['rcg'] for w in chunk_windows])  # [chunk_size, 50, 1, seq_len]
            ecg_chunk = np.stack([w['ecg'] for w in chunk_windows])  # [chunk_size, seq_len]
            pos_chunk = np.stack([w['positions'] for w in chunk_windows])  # [chunk_size, 50, 3]
            
            # Save chunk
            np.savez_compressed(
                os.path.join(self.output_dir, f'data_chunk_{chunk_idx:04d}.npz'),
                rcg=rcg_chunk,
                ecg=ecg_chunk,
                positions=pos_chunk
            )
            
            # Save chunk metadata
            chunk_metadata = [w['metadata'] for w in chunk_windows]
            with open(os.path.join(self.output_dir, f'metadata_chunk_{chunk_idx:04d}.json'), 'w') as f:
                json.dump(chunk_metadata, f, indent=2)
        
        print(f"Saved {len(windows)} windows in {num_chunks} chunks")
        
        # Create dataset info
        dataset_info = {
            'total_windows': len(windows),
            'seq_len': self.seq_len,
            'sample_step': self.sample_step,
            'num_subjects': len(set(m['subject_id'] for m in metadata)),
            'num_files': len(metadata),
            'chunk_size': chunk_size
        }
        
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def process_dataset(self):
        """Complete dataset processing pipeline"""
        windows, metadata = self.process_all_files()
        self.save_processed_data(windows, metadata)
        
        return len(windows)

class MMECGDataset(Dataset):
    """
    PyTorch Dataset for loading processed MMECG data
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.chunk_files = [f for f in os.listdir(data_dir) if f.startswith('data_chunk_') and f.endswith('.npz')]
        self.chunk_files.sort()
        
        # Load metadata
        self.metadata_files = [f for f in os.listdir(data_dir) if f.startswith('metadata_chunk_') and f.endswith('.json')]
        self.metadata_files.sort()
        
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
                
                # Convert to torch tensors
                rcg_tensor = torch.FloatTensor(rcg)
                ecg_tensor = torch.FloatTensor(ecg)
                positions_tensor = torch.FloatTensor(positions)
                
                return {
                    'rcg': rcg_tensor,
                    'ecg': ecg_tensor,
                    'positions': positions_tensor
                }
        
        raise IndexError(f"Index {idx} out of range")

def main():
    """Main function to process the dataset"""
    # Input and output directories
    input_dir = r"C:\Users\28474\Desktop\dataset\MMECG\finalPartialPublicData20221108"
    output_dir = r"C:\Users\28474\Desktop\dataset\MMECG\split"
    
    # Create processor
    processor = MMECGDataProcessor(input_dir, output_dir, seq_len=640, sample_step=30)
    
    # Process dataset
    total_windows = processor.process_dataset()
    
    print(f"Dataset processing complete!")
    print(f"Total windows created: {total_windows}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()