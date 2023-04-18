import torch
from torch import nn
import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding, LearnedFilterBank


class Model(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, embedding_size, 
                num_filters, hop_length, window_type='rectangular', # Learned filterbank parameters
                activation=['sigmoid', 'unit_norm']):
        super().__init__()
        
        self.representation = LearnedFilterBank(
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
        mix_repr = self.representation(data, direction='transform')
        
        data = self.amplitude_to_db(mix_repr)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        
        # Mask the mixture spectrogram
        estimates = mix_repr.unsqueeze(-1) * mask
        
        # Recombine estimates with mixture phase
        estimate_audio = self.representation(estimates, direction='inverse')
        
        return estimate_audio
