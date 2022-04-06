# PreTraining-ASR-From-Scratch - Instructions

Follow the steps below in order to PreTrain Wav2Vec2 from scrach using just the model configuration file


### Download LibriSpeech

In order to download LibriSpeech from OpenSLR.org, you can simply run the 'download_librispeech.py' script. 

The script takes two arguments when compiling:

    1. --data_dir_name
    2. --split
    
1. The name of the folder in which to download LibriSpeech. EX: '--data_dir_name Data' will create a directory called Data and download all the files in there
2. The split of LibriSpeech you would like to download. The available splits are 'dev', 'train-clean-100', 'train-clean-360', and 'test'. It is recommened to initially download 'train-clean-100' first, which is 100 hours of LibriSpeech cleaned audio. 


### Generate CSV File with Audio Metadata and Mappings to Audio Files



