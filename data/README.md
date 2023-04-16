Download MUSDB18HQ and place it in the data folder.
Torchaudio will be able to load the wav files.
```
└───MUSDB18HQ
    ├───test
    │   ├───Al James - Schoolboy Facination
    │   │       bass.wav
    │   │       drums.wav
    │   │       mixture.wav
    │   │       other.wav
    │   │       vocals.wav
    │   │
    ...
    └───train
        ├───A Classic Education - NightOwl
        │       bass.wav
        │       drums.wav
        │       mixture.wav
        │       other.wav
        │       vocals.wav
        ...
```

```
from torchaudio.datasets import MUSDB_HQ
train = MUSDB_HQ('data/', subset ='train')
test = MUSDB_HQ('data/', subset ='test')

train.names[0] # 'A Classic Education - NightOwl'
train.__len__() # 100
train.__getitem__(0)[0].size() # torch.Size([4, 2, 7560512])
```