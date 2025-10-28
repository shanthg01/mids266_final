"""
Training Pipeline for LSTMABAR
Includes data loading, training loop, and evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from lstmabar_model import LSTMABAR, LSTMABARTrainer
from musiccaps_loader import MusicCapsLoader


class MusicCapsDataset(Dataset):
    """
    PyTorch Dataset for MusicCaps with archetype annotations
    """
    
    def __init__(
        self,
        training_data_path: str,
        sample_rate: int = 44100,
        audio_duration: float = 2.0,
        augment: bool = False
    ):
        """
        Args:
            training_data_path: Path to .npz file with training data
            sample_rate: Audio sample rate
            audio_duration: Duration to load from each audio file
            augment: Whether to apply data augmentation
        """
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.augment = augment
        self.target_samples = int(sample_rate * audio_duration)
        
        # Load training data
        print(f"Loading training data from {training_data_path}")
        data = np.load(training_data_path, allow_pickle=True)
        
        self.archetype_vectors = torch.from_numpy(data['archetype_vectors']).float()
        self.descriptions = data['descriptions'].tolist()
        self.audio_paths = data['audio_paths'].tolist()
        self.archetype_order = data['archetype_order'].tolist()
        
        print(f"Loaded {len(self.descriptions)} training examples")
        
        # Filter out samples with missing audio files
        self.valid_indices = self._find_valid_samples()
        print(f"Found {len(self.valid_indices)} samples with valid audio files")
    
    def _find_valid_samples(self) -> List[int]:
        """Find indices with existing audio files"""
        valid = []
        for i, audio_path in enumerate(self.audio_paths):
            if Path(audio_path).exists():
                valid.append(i)
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample
        
        Returns:
            Dict with 'audio', 'description', 'archetype_weights'
        """
        # Map to valid index
        actual_idx = self.valid_indices[idx]
        
        # Load audio
        audio_path = self.audio_paths[actual_idx]
        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            duration=self.audio_duration
        )
        
        # Pad or trim to exact length
        if len(audio) < self.target_samples:
            audio = np.pad(audio, (0, self.target_samples - len(audio)))
        else:
            audio = audio[:self.target_samples]
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get description and archetype weights
        description = self.descriptions[actual_idx]
        archetype_weights = self.archetype_vectors[actual_idx]
        
        return {
            'audio': audio_tensor,
            'description': description,
            'archetype_weights': archetype_weights
        }
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply random audio augmentations"""
        # Random gain (±3dB)
        if np.random.random() > 0.5:
            gain_db = np.random.uniform(-3, 3)
            audio = audio * (10 ** (gain_db / 20))
        
        # Random time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-self.sample_rate // 10, self.sample_rate // 10)
            audio = np.roll(audio, shift)
        
        # Add slight noise
        if np.random.random() > 0.5:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        return audio


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader
    Handles variable-length descriptions
    """
    audio = torch.stack([item['audio'] for item in batch])
    descriptions = [item['description'] for item in batch]
    archetype_weights = torch.stack([item['archetype_weights'] for item in batch])
    
    return {
        'audio': audio,
        'descriptions': descriptions,
        'archetype_weights': archetype_weights
    }


class TrainingPipeline:
    """
    Complete training pipeline for LSTMABAR
    """
    
    def __init__(
        self,
        model: LSTMABAR,
        train_dataset: MusicCapsDataset,
        val_dataset: Optional[MusicCapsDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # Initialize trainer
        self.trainer = LSTMABARTrainer(
            model,
            learning_rate=learning_rate,
            loss_weights={
                'contrastive': 0.7,
                'archetype_prediction': 0.2,
                'audio_archetype_supervision': 0.1
            }
        )
        
        print(f"Training pipeline initialized:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Steps per epoch: {len(self.train_loader)}")
    
    def train(self):
        """Run complete training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Training
            train_losses = self.trainer.train_epoch(self.train_loader, epoch)
            print(f"\nTrain Losses: {train_losses}")
            
            # Validation
            if self.val_loader is not None:
                val_losses = {'total': 0.0}
                for batch in self.val_loader:
                    batch_losses = self.trainer.validate(
                        batch['descriptions'],
                        batch['audio'].to(self.model.device),
                        batch['archetype_weights'].to(self.model.device)
                    )
                    for key in batch_losses:
                        val_losses[key] = val_losses.get(key, 0.0) + batch_losses[key]
                
                # Average validation losses
                for key in val_losses:
                    val_losses[key] /= len(self.val_loader)
                
                print(f"Val Losses: {val_losses}")
                
                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    save_path = self.checkpoint_dir / 'best_model.pth'
                    self.model.save_checkpoint(
                        str(save_path),
                        epoch,
                        self.trainer.optimizer.state_dict()
                    )
                    print(f"✓ Best model saved (val_loss: {best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                self.model.save_checkpoint(
                    str(save_path),
                    epoch,
                    self.trainer.optimizer.state_dict()
                )
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        
        # Save final model
        final_path = self.checkpoint_dir / 'final_model.pth'
        self.model.save_checkpoint(
            str(final_path),
            self.num_epochs - 1,
            self.trainer.optimizer.state_dict()
        )
        
        # Plot training history
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training curves"""
        history = self.trainer.history
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        if history['val_loss']:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Component losses
        axes[1].plot(history['contrastive_loss'], label='Contrastive Loss')
        axes[1].plot(history['archetype_loss'], label='Archetype Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Component Losses')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=300)
        print(f"Training history plot saved to {self.checkpoint_dir / 'training_history.png'}")
        plt.close()


def prepare_musiccaps_data(
    csv_path: str,
    audio_dir: str = 'musiccaps_audio',
    output_path: str = 'musiccaps_training_data.npz',
    max_downloads: int = 500,
    train_split: float = 0.8
) -> Tuple[str, str]:
    """
    Prepare MusicCaps dataset for training
    
    Returns:
        Paths to train and val .npz files
    """
    print("=== Preparing MusicCaps Dataset ===")
    
    # Load and process MusicCaps
    loader = MusicCapsLoader(csv_path, audio_dir)
    
    # Download audio clips
    print(f"\nDownloading up to {max_downloads} audio clips...")
    downloaded, failed = loader.download_audio_clips(
        max_clips=max_downloads,
        use_balanced_subset=True
    )
    
    print(f"Successfully downloaded: {len(downloaded)}")
    print(f"Failed: {len(failed)}")
    
    # Create archetype training data
    print("\nCreating archetype training data...")
    training_data = loader.create_archetype_training_data(use_tfidf_weighting=True)
    
    # Split into train/val
    n_train = int(len(training_data) * train_split)
    train_data = training_data[:n_train]
    val_data = training_data[n_train:]
    
    # Save train set
    train_path = output_path.replace('.npz', '_train.npz')
    vectors_train = np.array([item['archetype_vector'] for item in train_data])
    descriptions_train = [item['description'] for item in train_data]
    audio_paths_train = [item['audio_path'] for item in train_data]
    
    np.savez_compressed(
        train_path,
        archetype_vectors=vectors_train,
        descriptions=descriptions_train,
        audio_paths=audio_paths_train,
        archetype_order=train_data[0]['archetype_order']
    )
    
    # Save val set
    val_path = output_path.replace('.npz', '_val.npz')
    vectors_val = np.array([item['archetype_vector'] for item in val_data])
    descriptions_val = [item['description'] for item in val_data]
    audio_paths_val = [item['audio_path'] for item in val_data]
    
    np.savez_compressed(
        val_path,
        archetype_vectors=vectors_val,
        descriptions=descriptions_val,
        audio_paths=audio_paths_val,
        archetype_order=val_data[0]['archetype_order']
    )
    
    print(f"\nDataset prepared:")
    print(f"  Train: {train_path} ({len(train_data)} samples)")
    print(f"  Val: {val_path} ({len(val_data)} samples)")
    
    return train_path, val_path


# Main training script
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Configuration
    config = {
        'embedding_dim': 768,
        'audio_architecture': 'resnet',  # or 'ast'
        'sample_rate': 44100,
        'audio_duration': 2.0,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'max_downloads': 500
    }
    
    # Step 1: Prepare dataset (run once)
    print("Step 1: Preparing MusicCaps dataset...")
    train_data_path, val_data_path = prepare_musiccaps_data(
        csv_path='musiccaps-public.csv',
        audio_dir='musiccaps_audio',
        max_downloads=config['max_downloads']
    )
    
    # Step 2: Create datasets
    print("\nStep 2: Creating PyTorch datasets...")
    train_dataset = MusicCapsDataset(
        train_data_path,
        sample_rate=config['sample_rate'],
        audio_duration=config['audio_duration'],
        augment=True
    )
    
    val_dataset = MusicCapsDataset(
        val_data_path,
        sample_rate=config['sample_rate'],
        audio_duration=config['audio_duration'],
        augment=False
    )
    
    # Step 3: Initialize model
    print("\nStep 3: Initializing LSTMABAR model...")
    model = LSTMABAR(
        embedding_dim=config['embedding_dim'],
        audio_architecture=config['audio_architecture'],
        sample_rate=config['sample_rate'],
        use_quantum_attention=False,
        device=device
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 4: Create training pipeline
    print("\nStep 4: Setting up training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        checkpoint_dir='checkpoints'
    )
    
    # Step 5: Train!
    print("\nStep 5: Starting training...")
    pipeline.train()
    
    print("\n✓ Training complete! Model checkpoints saved to checkpoints/")