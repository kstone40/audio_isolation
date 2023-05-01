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
from models.Waveform import Waveform

nussl.ml.register_module(MaskInference)
nussl.ml.register_module(UNetSpect)
nussl.ml.register_module(Filterbank)
nussl.ml.register_module(Waveform)

eval_list = ['mask_1source','mask_4source',
             'waveform_1source','waveform_4source',
             'filterbank_1source','filterbank_4source',
             'unet_1source','unet_4source'
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
    waveform_models = ['Filterbank','Waveform']
    if model_type in waveform_models:
        stft_params = None

        if configs['model_params']['num_sources']==1:
            tfm = nussl_tfm.Compose([
                nussl_tfm.SumSources([['bass', 'drums', 'other']]),
                nussl_tfm.GetAudio(),
                nussl_tfm.IndexSources('source_audio', 1),
                nussl_tfm.ToSeparationModel(),
            ])
        elif configs['model_params']['num_sources']==2:
            tfm = nussl_tfm.Compose([
                nussl_tfm.SumSources([['bass', 'drums', 'other']]),
                nussl_tfm.GetAudio(),
                nussl_tfm.ToSeparationModel(),
            ])
        elif configs['model_params']['num_sources']==4:
            tfm = nussl_tfm.Compose([
                nussl_tfm.GetAudio(),
                nussl_tfm.ToSeparationModel(),
            ])
        else:
            raise ValueError('Number of sources can only be 1 (vocals), 2 (vocals/accompaniement), or 4 (full sep)')

        target_key = 'source_audio'
        output_key = 'audio'
        input_key = 'mix_audio'

        separator = nussl.separation.deep.DeepAudioEstimation(
            nussl.AudioSignal(), model_path=model_path,
            device='cpu',
        )

    else:
        stft_params = nussl.STFTParams(**configs['stft_params'])

        if configs['model_params']['num_sources']==1:
            tfm = nussl_tfm.Compose([
                nussl_tfm.SumSources([['bass', 'drums', 'other']]),
                nussl_tfm.MagnitudeSpectrumApproximation(),
                nussl_tfm.IndexSources('source_magnitudes', 1),
                nussl_tfm.ToSeparationModel(),
            ])
        elif configs['model_params']['num_sources']==2:
            tfm = nussl_tfm.Compose([
                nussl_tfm.SumSources([['bass', 'drums', 'other']]),
                nussl_tfm.MagnitudeSpectrumApproximation(),
                nussl_tfm.ToSeparationModel(),
            ])
        elif configs['model_params']['num_sources']==4:
            tfm = nussl_tfm.Compose([
                nussl_tfm.MagnitudeSpectrumApproximation(),
                nussl_tfm.ToSeparationModel(),
            ])
        else:
            raise ValueError('Number of sources can only be 1 (vocals), 2 (vocals/accompaniement), or 4 (full sep)')

        target_key = 'source_magnitudes'
        output_key = 'estimates'
        input_key = 'mix_magnitude'

        separator = nussl.separation.deep.DeepMaskEstimation(
            nussl.AudioSignal(), model_path=model_path,
            device='cpu',
        )

    model_checkpoint = torch.load(model_path,map_location=torch.device(device))
    
    #Test on the data
    test_folder = configs['test_folder']
    tfm = None
    test_data = data.mixer(stft_params, transform=tfm, fg_path=configs['test_folder'], num_mixtures=test_iterations, coherent_prob=1.0, duration=test_duration)
    
    #Set up a master dictionary to combine the scores from each test data (per model)
    source_keys = list(test_data[0]['sources'].keys())
    all_scores = {}
    for source in source_keys:
        all_scores[source] = {}

    #Individually score each sample in the test data
    for i,item in enumerate(test_data):
        separator.audio_signal = item['mix']
        output = separator()

        source_keys = list(item['sources'].keys())

        if len(output)>1:
            order = ['bass','drums','other','vocals']
        else:
            order = ['vocals']
            
        estimates = []
        sources = []
        #print(item['sources'].keys())
        for j,source in enumerate(order[0:len(source_keys)+1]):
            estimates.append(output[j])
            sources.append(item['sources'][source])
        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=order
        )
        try:
            scores = evaluator.evaluate()
        except:
            print(f'Evaluation error with model {model_name} and sample {i}')
            continue
            
        for source in order:
            for score in scores[source]:
                if score not in all_scores[source].keys():
                    all_scores[source][score] = scores[source][score]
                else:
                    all_scores[source][score] += scores[source][score]
    
    #Record all metadata, and combine the scores with a mean (per model)
    for source in order:
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