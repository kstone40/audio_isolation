import torch
from torch import nn
import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding

nussl.utils.seed(0)
to_numpy = lambda x: x.detach().numpy()
to_tensor = lambda x: torch.from_numpy(x).reshape(-1, 1).float()

class Cluster(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size, num_layers, bidirectional, dropout,
                embedding_size, activation=['sigmoid', 'unit_norm']):
        super().__init__()
        
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
        
        mix_magnitude = data # save for masking
        
        data = self.amplitude_to_db(data)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        
        estimates = mix_magnitude.unsqueeze(-1) * mask
        
        return estimates