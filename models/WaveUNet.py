import nussl
from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super(WaveUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
    
        self.down_conv1 = WaveUNet.conv_block(self.in_channels, self.features)
        self.down_conv2 = WaveUNet.conv_block(self.features, self.features*2)
        self.down_conv3 = WaveUNet.conv_block(self.features*2, self.features*4)
        self.down_conv4 = WaveUNet.conv_block(self.features*4, self.features*8)
        self.down_conv5 = WaveUNet.conv_block(self.features*8, self.features*16)
        
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.transpose1 = nn.ConvTranspose1d(self.features*16, self.features*8, kernel_size=2, stride=2)
        self.up_conv1 = WaveUNet.conv_block(self.features*16, self.features*8)
        self.transpose2 = nn.ConvTranspose1d(self.features*8, self.features*4, kernel_size=2, stride=2, output_padding=1)
        self.up_conv2 = WaveUNet.conv_block(self.features*8, self.features*4)
        self.transpose3 = nn.ConvTranspose1d(self.features*4, self.features*2, kernel_size=2, stride=2)
        self.up_conv3 = WaveUNet.conv_block(self.features*4, self.features*2)
        self.transpose4 = nn.ConvTranspose1d(self.features*2, self.features, kernel_size=2, stride=2)
        self.up_conv4 = WaveUNet.conv_block(self.features*2, self.features)
        
        self.out = nn.Conv1d(self.features, self.out_channels, kernel_size=1)

    def forward(self, data):
        
        # mix_magnitude = data.unsqueeze(4) # save for masking
        # data = data.transpose(3, 1).transpose(2, 3)
        
        down_conv1 = self.down_conv1(data)
        down_conv1_max = self.max_pool(down_conv1)
        down_conv2 = self.down_conv2(down_conv1_max)
        down_conv2_max = self.max_pool(down_conv2)
        down_conv3 = self.down_conv3(down_conv2_max)
        down_conv3_max = self.max_pool(down_conv3)
        down_conv4 = self.down_conv4(down_conv3_max)
        down_conv4_max = self.max_pool(down_conv4)
        
        down_conv5 = self.down_conv5(down_conv4_max)

        trans1 = self.transpose1(down_conv5)
        up_conv1 = self.up_conv1(torch.cat([down_conv4, trans1], 1))
        trans2 = self.transpose2(up_conv1)
        up_conv2 = self.up_conv2(torch.cat([down_conv3, trans2], 1))
        trans3 = self.transpose3(up_conv2)
        up_conv3 = self.up_conv3(torch.cat([down_conv2, trans3], 1))
        trans4 = self.transpose4(up_conv3)
        up_conv4 = self.up_conv4(torch.cat([down_conv1, trans4], 1))
        
        estimates = self.out(up_conv4).unsqueeze(3)
        #mask = mask.transpose(1, 3).transpose(1, 2).unsqueeze(4)
        
        # estimates = mix_magnitude * mask
        
        output = {
            'estimates': estimates
        }
        
        return output

    @staticmethod
    def conv_block(in_channels, features):
        
        conv = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True)
        )
        
        return conv
    
    # Added function
    @classmethod
    def build(cls, in_channels=1, out_channels=1, features=16):
        # Step 1. Register our model with nussl
        nussl.ml.register_module(cls)
        
        # Step 2a: Define the building blocks.
        modules = {
            'model': {
                'class': 'WaveUNet',
                'args': {
                    # 'num_features': num_features,
                    # 'num_audio_channels': num_audio_channels,
                    # 'hidden_size': hidden_size,
                    # 'num_layers': num_layers,
                    # 'bidirectional': bidirectional,
                    # 'dropout': dropout,
                    # 'num_sources': num_sources,
                    # 'activation': activation
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'features': features
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
        for key in ['estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        # Step 2d. There are two outputs from our SeparationModel: estimates and mask.
        # Then put it all together.
        output = ['estimates']
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        # Step 3. Instantiate the model as a SeparationModel.
        return nussl.ml.SeparationModel(config)