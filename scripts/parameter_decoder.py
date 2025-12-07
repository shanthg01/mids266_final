import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterDecoder(nn.Module):
    """
    Expands Macro-Controls into granular DDSP parameters.
    Input: 32 Latents
    Output: Harmonic Amplitudes (64), Noise Bands (32), ADSR (4)
    """
    def __init__(self, input_dim=32, n_harmonics=64, n_noise_bands=32, device='cpu'):
        super().__init__()
        self.device = device
        self.n_harmonics = n_harmonics
        
        # Decoder Network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        
        # --- Granular Output Heads ---
        
        # 1. Harmonic Distribution (Amplitudes for 64 harmonics)
        # Sigmoid -> Amplitudes are 0 to 1
        self.harmonic_head = nn.Linear(256, n_harmonics)
        
        # 2. Noise Filter Bands (Magnitudes for filtered noise)
        self.noise_head = nn.Linear(256, n_noise_bands)
        
        # 3. Explicit ADSR (The only strictly physics-based controls we keep distinct)
        self.adsr_head = nn.Linear(256, 4) 
        
        # 4. Global Gain
        self.gain_head = nn.Linear(256, 1)

        self.to(device)

    def forward(self, latents):
        # latents shape: (Batch, 32)
        x = self.net(latents)
        
        # 1. Harmonics: Normalize so they sum to 1 (Distribution) then scale by gain
        raw_harmonics = torch.sigmoid(self.harmonic_head(x))
        # Softmax ensures we predict a spectral *shape*, not just loudness
        harmonic_dist = F.softmax(raw_harmonics, dim=1) 
        
        # 2. Noise: Filter bands (0 to 1)
        noise_bands = torch.sigmoid(self.noise_head(x))
        
        # 3. ADSR: Sigmoid (0 to 1)
        adsr = torch.sigmoid(self.adsr_head(x))
        
        # 4. Global Amplitude
        gain = torch.sigmoid(self.gain_head(x))
        
        return harmonic_dist, noise_bands, adsr, gain