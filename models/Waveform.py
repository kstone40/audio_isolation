import torch
from torch import nn
import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding, STFT

class Waveform(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, embedding_size, 
                num_filters, hop_length, window_type='sqrt_hann', # New STFT parameters
                activation=['sigmoid', 'unit_norm']):
        super().__init__()
        
        self.stft = STFT(
            num_filters, 
            hop_length=hop_length, 
            window_type=window_type
        )
        
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                embedding_size, activation, 
                                num_audio_channels)
        
    def forward(self, data):
        # Take STFT inside model
        mix_stft = self.stft(data, direction='transform')
        nb, nt, nf, nac = mix_stft.shape
        # Stack the mag/phase along the second to last axis
        mix_stft = mix_stft.reshape(nb, nt, 2, -1, nac)
        mix_magnitude = mix_stft[:, :, 0, ...] # save for masking
        mix_phase = mix_stft[:, :, 1, ...] # save for reconstruction
        
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        
        # Mask the mixture spectrogram
        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        # Recombine estimates with mixture phase
        mix_phase = mix_phase.unsqueeze(-1).expand_as(estimates)
        estimates = torch.cat([estimates, mix_phase], dim=2)
        estimate_audio = self.stft(estimates, direction='inverse')
        
        return estimate_audio
