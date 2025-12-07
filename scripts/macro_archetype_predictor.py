import torch
import torch.nn as nn
import torch.nn.functional as F

class MacroArchetypePredictor(nn.Module):
    """
    Predicts abstract Macro-Controls rather than raw physical parameters.
    Output: 32-dimensional latent vector split into semantic groups.
    """
    def __init__(self, embedding_dim=768, hidden_dims=[512, 256], device='cpu'):
        super().__init__()
        self.device = device
        
        # Shared Feature Extraction
        layers = []
        input_dim = embedding_dim * 2 # Text + Audio embeddings
        for h in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            input_dim = h
        self.shared = nn.Sequential(*layers)
        
        # --- Macro Heads (The Latent Space) ---
        
        # 1. Timbre Latents (16 dims): Represents spectral shape/color
        self.timbre_head = nn.Linear(input_dim, 16)
        
        # 2. Envelope Latents (8 dims): Represents temporal evolution curve
        self.envelope_head = nn.Linear(input_dim, 8)
        
        # 3. Modulation Latents (8 dims): Represents movement/texture
        self.modulation_head = nn.Linear(input_dim, 8)
        
        self.to(device)
    
    def _ensure_tensor(self, x):
        """Helper to safely extract tensor from tuple"""
        if isinstance(x, tuple):
            return x[0]
        return x

    def forward(self, text_emb, audio_emb):
        # 1. Unpack tuples if necessary (Robustness fix)
        text_emb = self._ensure_tensor(text_emb)
        audio_emb = self._ensure_tensor(audio_emb)
        
        # 2. Squeeze extra dimensions if present (e.g. [1, 1, 768] -> [1, 768])
        # We check .dim() only after ensuring it's a tensor
        if text_emb.dim() == 3: text_emb = text_emb.squeeze(1)
        if audio_emb.dim() == 3: audio_emb = audio_emb.squeeze(1)
            
        # 3. Concatenate
        joint = torch.cat([text_emb, audio_emb], dim=1)
        features = self.shared(joint)
        
        # 4. Predict Macros (Unbounded latent space, usually Tanh or raw)
        # We use Tanh to keep latents between -1 and 1 for stability
        timbre_z = torch.tanh(self.timbre_head(features))
        env_z = torch.tanh(self.envelope_head(features))
        mod_z = torch.tanh(self.modulation_head(features))
        
        # Return dictionary for interpretability, or concatenated for storage
        return {
            "timbre": timbre_z,
            "envelope": env_z,
            "modulation": mod_z,
            "combined": torch.cat([timbre_z, env_z, mod_z], dim=1)
        }