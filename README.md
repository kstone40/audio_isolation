# audio_isolation
GA Tech CS7643 Spring 2023 course project: Isolation of instruments/voices from music or other audio


## Installation
With conda installed, open a terminal and run the command:
```
conda env create --file environment-mac.yml
```
**Note:** You will need to download the HQ musdb18 dataset and place a sample in the data folder to run the notebook. We created a directory `/data`, and used the functions `prepare_musdb()` and `prepare_musdbhq()` from the data module, provided by https://github.com/source-separation/tutorial.
