import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import torch

class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        
        super().__init__()
        
        #Scale spectrograms to dB range (logscale)
        self.amplitude_to_db = AmplitudeToDB()
        
        #Batch norm
        self.input_normalization = BatchNorm(num_features)
        
        #N layers of BLSTM
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        
        #FC layers to calculate mask (or embedding)
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, num_sources, activation, num_audio_channels)
        
    def forward(self, data):
        mix_magnitude = data # save for masking
        
       
        #Scale spectrograms to dB range (logscale)
        data = self.amplitude_to_db(mix_magnitude)
        
        #Batch norm
        data = self.input_normalization(data)
        
        #N layers of BLSTM
        data = self.recurrent_stack(data)
        
        #FC layers to calculate mask (or embedding)
        mask = self.embedding(data)
        mix_magnitude = mix_magnitude.unsqueeze(-1)
        
        #print(mix_magnitude.shape)
        #print(mask.shape)
        
        estimates = mix_magnitude * mask
        
        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output
    
    # Added function
    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size, 
            num_layers, bidirectional, dropout, num_sources, 
            activation='sigmoid'):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(cls)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }
        
        # Step 2b: Define the connections between input and output.
        # Here, the mix_magnitude key is the only input to the model.
        connections = [
            ['model', ['mix_magnitude']]
        ]
        
        # Step 2c. The model outputs a dictionary, which SeparationModel will
        # change the keys to model:mask, model:estimates. The lines below 
        # alias model:mask to just mask, and model:estimates to estimates.
        # This will be important later when we actually deploy our model.
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates', 'mask',]
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)
