import torch
from torch import nn
import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding, LearnedFilterBank


class Filterbank(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, num_sources, 
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
                                num_sources, activation, 
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
        
        output = {
            #'mask': mask,
            #'estimates': estimate_audio
            'audio': estimate_audio
        }
        
        return output

    # Added function
    @classmethod
    def build(cls,  num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, num_sources, 
                num_filters, hop_length, window_type='rectangular', # Learned filterbank parameters
                activation=['sigmoid', 'unit_norm']):
        
        # Step 1. Register our model with nussl
        nussl.ml.register_module(cls)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'Filterbank',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'num_filters': num_filters,
                    'hop_length':hop_length,
                    'window_type':window_type,
                    'activation':activation
                }
            }
        }
        
        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['mix_audio']]
        ]
        
        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below 
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['audio']: #['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['audio'] #['estimates', 'mask',]
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)
