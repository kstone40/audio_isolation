import torch
import nussl
from nussl.datasets import transforms as nussl_tfm
from models.MaskInference import MaskInference
from models.UNet import UNetSpect
from models.Filterbank import Filterbank
from models.Waveform import Waveform
from utils import utils, data
from pathlib import Path
import yaml, argparse

#Set up configuration from optional command line --config argument, else default
global args
parser = argparse.ArgumentParser(description='DL Source Separation')
parser.add_argument('--config', default='config/mask_default.yml')
args = parser.parse_args()

#Load yaml configs into configs dictionary
with open(args.config,'r') as f:
    configs = yaml.safe_load(f)
    f.close()
    
utils.logger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = configs['model_type']
model_dict = {'Mask': MaskInference,
              'UNet': UNetSpect,
              'Filterbank':Filterbank,
              'Waveform':Waveform,
             }
waveform_models = ['Filterbank','Waveform','WaveUNet']
assert model_type in model_dict.keys(), f'Model type must be one of {model_dict.keys()}'

if model_type in waveform_models:
    stft_params = None
    
    tfm = nussl_tfm.Compose([
        #nussl_tfm.SumSources([['bass', 'drums', 'other']]),
        nussl_tfm.GetAudio(),
        #nussl_tfm.IndexSources('source_audio', 1),
        nussl_tfm.ToSeparationModel(),
    ])
    
    target_key = 'source_audio'
    output_key = 'audio'
    
else:
    stft_params = nussl.STFTParams(**configs['stft_params'])
    
    tfm = nussl_tfm.Compose([
        #nussl_tfm.SumSources([['bass', 'drums', 'other']]),
        nussl_tfm.MagnitudeSpectrumApproximation(),
        #nussl_tfm.IndexSources('source_magnitudes', 1),
        nussl_tfm.ToSeparationModel(),
    ])
    
    target_key = 'source_magnitudes'
    output_key = 'estimates'


configs['batch_size'] = 1
configs['train_generator_params']['num_mixtures']=10
configs['valid_generator_params']['num_mixtures']=1

duration=5

train_data = data.on_the_fly(stft_params, transform=tfm, fg_path=configs['test_folder'], **configs['train_generator_params'], duration=duration)
train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=1, batch_size=configs['batch_size'])

val_data = data.on_the_fly(stft_params, transform=tfm, fg_path=configs['test_folder'], **configs['valid_generator_params'], duration=duration)
val_dataloader = torch.utils.data.DataLoader(val_data, num_workers=1, batch_size=configs['batch_size'])

overfit_selection=1

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
    loss = loss_fn(output[output_key],batch[target_key])
    
    #Backward pass
    loss.backward()
    optimizer.step()
    
    loss_vals = {'loss':loss.item()}
    
    return loss_vals

def val_step(engine, batch):
    with torch.no_grad():
        output = model(batch)
    loss = loss_fn(output[output_key],batch[target_key])  
    loss_vals = {'loss':loss.item()}
    return loss_vals

#Set up the model and optimizer
if model_type=='Mask':
    model = model_dict[model_type].build(num_features=stft_params.window_length//2+1,**configs['model_params']).to(device)
elif model_type == 'Waveform':
    model = model_dict[model_type].build(num_features=configs['model_params']['num_filters']//2+1,**configs['model_params']).to(device)
else:
    model = model_dict[model_type].build(**configs['model_params']).to(device)

optimizer = torch.optim.Adam(model.parameters(), **configs['optimizer_params'])

for i,batch in enumerate(train_dataloader):
    if i==overfit_selection:
        batch=batch
        break
    
for key in batch:
    if torch.is_tensor(batch[key]):
        batch[key] = batch[key].float().to(device)   
        
optimizer = torch.optim.Adam(model.parameters(), **configs['optimizer_params'])

# Create nussl ML engine
trainer, validator = nussl.ml.train.create_train_and_validation_engines(train_step, val_step, device=device)

N_ITERATIONS = 1000 #1000
loss_history = [] # For bookkeeping

for i in range(N_ITERATIONS):
    loss_val = train_step(trainer,batch)
    loss_history.append(loss_val['loss'])
    if i%100==0:
        print(f'Loss: {loss_val["loss"]:.6f} at iteration {i}')
        
configs['train_params']['epoch_length']=2
configs['train_params']['max_epochs']=1
configs['optimizer_params']['lr'] = 1e-15
optimizer = torch.optim.Adam(model.parameters(), **configs['optimizer_params'])

# Save model outputs
checkpoint_folder = Path('models/overfit').absolute()

# Adding handlers from nussl that print out details about model training
# run the validation step, and save the models.
nussl.ml.train.add_stdout_handler(trainer, validator)
nussl.ml.train.add_validate_and_checkpoint(checkpoint_folder, model, optimizer, train_data, trainer, val_dataloader, validator)
nussl.ml.train.add_progress_bar_handler(trainer, validator)

with open('models/overfit/configs.yml', 'w') as file:
    yaml.dump(configs, file)

trainer.run(train_dataloader, **configs['train_params'])


