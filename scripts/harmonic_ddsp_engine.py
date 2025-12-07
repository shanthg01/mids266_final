import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HarmonicDDSPEngine(nn.Module):
    """
    High-Fidelity Generator.
    Uses Additive Synthesis (Sum of Sinusoids) + Subtractive Noise.
    """
    def __init__(self, sample_rate=44100, n_harmonics=64, device='cpu'):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.device = device
        
    def generate_adsr(self, num_samples, attack, decay, sustain, release):
        # (Same robust ADSR implementation as before)
        a_samp = (attack * 0.5 * self.sample_rate).long() + 1
        d_samp = (decay * 0.5 * self.sample_rate).long() + 1
        r_samp = (release * 0.5 * self.sample_rate).long() + 1
        
        total_len = a_samp + d_samp + r_samp
        if total_len > num_samples:
            scale = num_samples / total_len
            a_samp = (a_samp * scale).long()
            d_samp = (d_samp * scale).long()
            r_samp = (r_samp * scale).long()
            
        s_samp = num_samples - (a_samp + d_samp + r_samp)
        if s_samp < 0: s_samp = 0
            
        t = torch.arange(num_samples, device=self.device).float()
        
        # Construct envelope segments
        attack_curve = torch.linspace(0, 1, a_samp, device=self.device)
        decay_curve = torch.linspace(1, sustain, d_samp, device=self.device)
        sustain_curve = torch.full((s_samp,), sustain, device=self.device)
        release_curve = torch.linspace(sustain, 0, r_samp, device=self.device)
        
        env = torch.cat([attack_curve, decay_curve, sustain_curve, release_curve], dim=0)
        return F.pad(env, (0, num_samples - len(env)))[:num_samples]

    def forward(self, base_audio, harmonic_dist, noise_bands, adsr, gain):
        """
        base_audio: (Batch, Samples) - used for duration reference
        harmonic_dist: (Batch, n_harmonics)
        noise_bands: (Batch, n_bands)
        adsr: (Batch, 4) [A, D, S, R]
        gain: (Batch, 1)
        """
        batch_size = base_audio.shape[0]
        num_samples = base_audio.shape[-1]
        
        # Time vector
        t = torch.linspace(0, num_samples/self.sample_rate, num_samples, device=self.device)
        t = t.unsqueeze(0).expand(batch_size, -1) # (Batch, Samples)
        
        # --- 1. Additive Synthesis ---
        # Fundamental Frequency (fixed at A4=440Hz for timbre transfer demo)
        f0 = 440.0
        
        # Generate Harmonics
        # We sum: Amplitude[k] * sin(2 * pi * k * f0 * t)
        signal = torch.zeros_like(base_audio, device=self.device)
        
        # Vectorized harmonic summation
        # k ranges from 1 to n_harmonics
        k = torch.arange(1, self.n_harmonics + 1, device=self.device).view(1, -1, 1) # (1, Harmonics, 1)
        
        # Phase: (Batch, Harmonics, Samples)
        # 2 * pi * k * f0 * t
        phases = 2 * np.pi * k * f0 * t.unsqueeze(1) 
        
        # Amplitudes: (Batch, Harmonics, 1)
        # harmonic_dist comes in as (Batch, Harmonics)
        amplitudes = harmonic_dist.unsqueeze(2) 
        
        # Summation: Sum over Harmonic dimension
        harmonics = torch.sum(amplitudes * torch.sin(phases), dim=1)
        
        # --- 2. Noise Synthesis ---
        # Generate white noise
        noise = torch.rand_like(base_audio, device=self.device) * 2 - 1
        
        # Apply "Noise Color" via bands (Simplified: Weighted average of noise magnitude)
        # In a full implementation, this would be a filter bank. 
        # Here we approximate by scaling the noise amplitude based on the predicted noise level.
        noise_level = torch.mean(noise_bands, dim=1, keepdim=True) * 0.1 # Scale down noise
        noise = noise * noise_level
        
        # --- 3. Enveloping & Mixing ---
        final_signal = harmonics + noise
        
        # Apply ADSR
        envelopes = []
        for i in range(batch_size):
            env = self.generate_adsr(num_samples, adsr[i,0], adsr[i,1], adsr[i,2], adsr[i,3])
            envelopes.append(env)
        envelopes = torch.stack(envelopes)
        
        # Apply Gain and Envelope
        final_signal = final_signal * envelopes * gain
        
        # Normalize
        max_val = torch.max(torch.abs(final_signal), dim=1, keepdim=True)[0] + 1e-5
        return final_signal / max_val