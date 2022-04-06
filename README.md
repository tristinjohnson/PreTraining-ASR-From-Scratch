# PreTraining ASR From Scratch using Wav2Vec2.0

Follow the steps below in order to PreTrain Wav2Vec2 from scrach using just the model configuration file


## Download LibriSpeech

In order to download LibriSpeech from OpenSLR.org, you can simply run the 'download_librispeech.py' script in the 'generate_audio_mappings' directory. 

The script takes two arguments when compiling:

    1. --data_dir_name
    2. --split
    
1. The name of the folder in which to download LibriSpeech. EX: '--data_dir_name Data' will create a directory called Data and download all the files in there
2. The split of LibriSpeech you would like to download. The available splits are 'dev', 'train-clean-100', 'train-clean-360', and 'test'. It is recommened to initially download 'train-clean-100' first, which is 100 hours of LibriSpeech cleaned audio. 


## Generate CSV File with Audio Metadata and Mappings to Audio Files

The training file only works when reading metadata about the audio files from a CSV file. In order to create this CSV file, simply run the 'generate_audio_paths_excel.py' script in the 'generate_audio_mappings' directory with the following arguments:

    1. --path
    2. --csv_name
    
1. Provide the full path to the data directory where the data is stored. EX: '--path /home/ubuntu/project/Data/LibriSpeech/dev-clean'
2. Provide any name you would like the CSV file to be named. It is recommended you name the CSV file based on whichever split you download from the previous step with the '.csv' extension at the end. EX: if you download 'train-clean-100', name your file something like 'librispeech_train_100.csv'

After running the script, the CSV file will be saved in the same directory as the 'generate_audio_paths_excel.py'


## Pre-Train Wav2Vec2 from Scratch

Once you have successfully completed the previous two steps, you can now begin pretraining Wav2Vec2.0 from scratch using PyTorch. In order to start pretraining, you can run the 'librispeech_pytorch_from_scratch.py' script in the 'pretraining_from_scratch' directory with the following arguments:

    1. --batch_size
    2. --num_epochs
    3. --path_to_csv
    4. --num_training_samples
    
1. 

