import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nussl
from utils import viz, data
import yaml
import os
import json
from nussl.datasets import transforms as nussl_tfm
from pathlib import Path
import warnings
warnings.simplefilter("ignore")

from models.MaskInference import MaskInference
from models.UNet import UNetSpect
from models.Filterbank import Filterbank
nussl.ml.register_module(MaskInference)
nussl.ml.register_module(UNetSpect)
nussl.ml.register_module(Filterbank)

eval_list = [
             'ST_mask_tutorial_defaults',
             'ST_mask_1layer','ST_mask_3layer','ST_mask_5layer',
             'ST_mask_0.1dropout','ST_mask_0.5dropout',
            ]

test_iterations = 50 #number of samples
test_duration = 5 #seconds

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device} for evaluation')

data_log = []

for model_name in eval_list:
    print('Evaluating model: ' + model_name)
    #Load the model and config
    model_path = 'models/'+model_name+'/checkpoints/latest.model.pth'
    config_path = 'models/'+model_name+'/configs.yml'
    
   #Load yaml configs into configs dictionary
    with open(config_path,'r') as f:
        configs = yaml.safe_load(f)
        f.close()

    model_type = configs['model_type']
    waveform_models = ['Filterbank']
    if model_type in waveform_models:
        stft_params = None

        separator = nussl.separation.deep.DeepAudioEstimation(
            nussl.AudioSignal(), model_path='overfit/checkpoints/latest.model.pth',
            device='cpu',
        )
    else:
        stft_params = nussl.STFTParams(**configs['stft_params'])

        separator = nussl.separation.deep.DeepMaskEstimation(
            nussl.AudioSignal(), model_path=model_path,
            device='cpu',
        )

    model_checkpoint = torch.load(model_path,map_location=torch.device(device))
    
    #Test on the data
    test_folder = configs['test_folder']
    tfm = nussl_tfm.Compose([
        nussl_tfm.SumSources([['bass', 'drums', 'other']]),
    ])
    test_data = data.mixer(stft_params, transform=tfm, fg_path=configs['test_folder'], num_mixtures=test_iterations, coherent_prob=1.0, duration=test_duration)
    
    #Set up a master dictionary to combine the scores from each test data (per model)
    source_keys = list(test_data[0]['sources'].keys())
    all_scores = {}
    for source in source_keys:
        all_scores[source] = {}

    #Individually score each sample in the test data
    for i,item in enumerate(test_data):
        separator.audio_signal = item['mix']
        estimates = separator()

        source_keys = list(item['sources'].keys())
        estimates = {
            'vocals': estimates[0],
            'bass+drums+other': item['mix'] - estimates[0]
        }

        sources = [item['sources'][k] for k in source_keys]
        estimates = [estimates[k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        try:
            scores = evaluator.evaluate()
        except:
            print(f'Evaluation error with model {model_name} and sample {i}')
            continue
            
        for source in source_keys:
            for score in scores[source]:
                if score not in all_scores[source].keys():
                    all_scores[source][score] = scores[source][score]
                else:
                    all_scores[source][score] += scores[source][score]
    
    #Record all metadata, and combine the scores with a mean (per model)
    for source in source_keys:
        row = {'Source':source, 'Model':configs['model_type'], 'Loss Type':configs['loss_type'], 'Final Loss':model_checkpoint['metadata']['trainer.state_dict']['output']['loss']}
        if 'stft_params' in configs.keys():
            if configs['stft_params'] is not None:
                for stft_param in configs['stft_params'].keys():
                    row['STFT '+stft_param] = configs['stft_params'][stft_param]
        for model_param in configs['model_params'].keys():
            row['Model '+model_param] = configs['model_params'][model_param]
        for optimizer_param in configs['optimizer_params'].keys():
            row['Optimizer '+optimizer_param] = configs['optimizer_params'][optimizer_param]
        row.update({'Epochs':configs['train_params']['max_epochs'],
                    'Epoch Length':configs['train_params']['epoch_length'],
                    'Train Coherent Fraction':configs['train_generator_params']['coherent_prob'],
                    'Batch Size': configs['batch_size'],
                    'Test Size':test_iterations,
                    'Test Length':test_duration
                   })
        for score in all_scores[source]:
            row[score] = np.array(all_scores[source][score]).mean()
        data_log.append(row)

df_results = pd.DataFrame(data_log)

df_results.to_csv('Model_Evaluations.csv')