import torch
from torch import nn
import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding, STFT


class Waveform(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, num_sources, 
                num_filters, hop_length, window_type='sqrt_hann', 
                activation=['sigmoid']):
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
        self.embedding = Embedding(num_features, hidden_size, num_sources, activation, num_audio_channels)
        
    def forward(self, data):
        # Take STFT inside model
        mix_stft = self.stft(data, direction='transform')
        nb, nt, nf, nac = mix_stft.shape

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
        
        output = {
            'audio': estimate_audio
        }
        
        return output

    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, num_sources, 
                num_filters, hop_length, window_type='sqrt_hann', # New STFT parameters
                activation='sigmoid'):
        
        # Step 1. Register our model with nussl
        nussl.ml.register_module(cls)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'Waveform',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'num_filters': num_filters,
                    'hop_length': hop_length,
                    'window_type': window_type,
                    'activation': activation
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
        for key in ['audio']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        # modules['estimates'] = {'class': 'Alias'}
        # connections.append(['estimates', 'model: estimates'])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['audio']
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)


#nussl.ml.register_module(Waveform)