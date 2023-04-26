# import os
# print(os.environ['CONDA_DEFAULT_ENV'])
# import sys
# print(sys.version)

import torch
import nussl
from nussl.datasets import transforms as nussl_tfm
from models.MaskInference import MaskInference
from models.UNet import UNetSpect
from utils import utils, data
from pathlib import Path
import yaml, argparse

#Set up configuration from optional command line --config argument, else default
global args
parser = argparse.ArgumentParser(description='DL Source Separation')
parser.add_argument('--config', default='config/test_auto.yml')
args = parser.parse_args()

#Load yaml configs into configs dictionary
with open(args.config,'r') as f:
    configs = yaml.safe_load(f)
    f.close()
    
utils.logger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

stft_params = nussl.STFTParams(**configs['stft_params'])

#############CHANGE FOR WAVEFORM##################
tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    nussl_tfm.MagnitudeSpectrumApproximation(),
    nussl_tfm.IndexSources('source_magnitudes', 1),
    nussl_tfm.ToSeparationModel(),
])

#############CHANGE FOR WAVEFORM##################
train_data = data.on_the_fly(stft_params, transform=tfm, fg_path=configs['test_folder'], **configs['train_generator_params'])
train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=1, batch_size=configs['batch_size'])

val_data = data.on_the_fly(stft_params, transform=tfm, fg_path=configs['valid_folder'], **configs['valid_generator_params'])
val_dataloader = torch.utils.data.DataLoader(val_data, num_workers=1, batch_size=configs['batch_size'])

loss_type = configs['loss_type']
loss_dict = {'L1': nussl.ml.train.loss.L1Loss,
             'L2': nussl.ml.train.loss.MSELoss,
             'MSE': nussl.ml.train.loss.MSELoss,}
assert loss_type in loss_dict.keys(), f'Loss type must be one of {loss_dict.keys()}'
loss_fn = loss_dict[loss_type]()

def train_step(engine, batch):
    optimizer.zero_grad()
    
    #Forward pass
    output = model(batch)
    loss = loss_fn(output['estimates'],batch['source_magnitudes'])
    
    #Backward pass
    loss.backward()
    optimizer.step()
    
    loss_vals = {'loss':loss.item()}
    
    return loss_vals

def val_step(engine, batch):
    with torch.no_grad():
        output = model(batch)
    loss = loss_fn(output['estimates'],batch['source_magnitudes'])  
    loss_vals = {'loss':loss.item()}
    return loss_vals

model_type = configs['model_type']
model_dict = {'Mask': MaskInference,
              'UNet': UNetSpect}
assert model_type in model_dict.keys(), f'Model type must be one of {model_dict.keys()}'
#Set up the model and optimizer
if model_type=='Mask':
    model = MaskInference.build(stft_params.window_length//2+1, **configs['model_params']).to(device)
elif model_type=='UNet':
    model = UNetSpect.build(**configs['model_params']).to(device)
    
optimizer = torch.optim.Adam(model.parameters(), **configs['optimizer_params'])

# Create nussl ML engine
trainer, validator = nussl.ml.train.create_train_and_validation_engines(train_step, val_step, device=device)

# Save model outputs
checkpoint_folder = Path('models/'+configs['save_name']).absolute()

# Adding handlers from nussl that print out details about model training
# run the validation step, and save the models.
nussl.ml.train.add_stdout_handler(trainer, validator)
nussl.ml.train.add_validate_and_checkpoint(checkpoint_folder, model, optimizer, train_data, trainer, val_dataloader, validator)
nussl.ml.train.add_progress_bar_handler(trainer, validator)

import os
path = 'models/'+configs['save_name']
if not os.path.exists(path):
    os.makedirs(path)

with open('models/'+configs['save_name']+'/configs.yml', 'w') as file:
    yaml.dump(configs, file)

trainer.run(train_dataloader, **configs['train_params'])
