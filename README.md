# audio_isolation
GA Tech CS7643 Spring 2023 course project: Isolation of instruments/voices from music or other audio


## Installation
#### FFmpeg
Torchaudio I/O operations are dependent on [ffmpeg](https://ffmpeg.org/about.html), the leading multimedia framework used for encoding/decoding audio and video files.  Currently, Torchaudio works with ffmpeg version (>=4.1, <4.4).  Here are the steps to install ffmpeg on windows.
1. Download the already compiled ffmpeg executable file titled [ffmpeg-n4.4-latest-win64-lgpl-4.4.zip](https://github.com/BtbN/FFmpeg-Builds/releases).
   
   Note: I could not find version 4.3. 4.4 worked fine.
2. Unzip the file, rename it FFmpeg, and add it to root directory.
3. Click on the windows icon in the system tray, search edit environment variables, and click on it.
4. Click on Environment Variables and double click the Path variable.
5. Add C:\FFmpeg\bin and save.
6. Restart any open terminal and type `ffmpeg` to verify the installation.

#### Conda Environment
Next, install the conda environment.  Conda is recommended over virtualenv since there are packages on the conda-forge repository missing from PyPI.  Open the Anaconda Prompt and run `conda env create -n <ENVNAME> --file environment.yml` replacing ENVNAME with a name of your choice.  I am using python 3.10.10 if you run into any issues.  I believe I read somewhere in the Torchaudio documentation that 3.11 is not compatible yet.  Verify the install was correct by running `jupyter lab` and running all cells in demo.ipynb.  One song from musdb18 is provided in data/demo directory.
