# PreTraining ASR From Scratch using Wav2Vec2.0

Follow the steps below in order to PreTrain Wav2Vec2 from scrach using just the model configuration file


## Download LibriSpeech

In order to download LibriSpeech from OpenSLR.org, you can simply run the 'download_librispeech.py' script in the 'generate_audio_mappings' directory. 

The script takes two arguments when compiling:

    python3 download_librispeech.py --data_dir_name DATA_DIR_NAME --split SPLIT
    
    # example
    python3 download_librispeech.py --data_dir_name Data --split train-clean-100
    
1. --data_dir_name: The name of the folder in which to download LibriSpeech. EX: '--data_dir_name Data' will create a directory called Data and download all the files in there
2. --split: The split of LibriSpeech you would like to download. The available splits are 'dev', 'train-clean-100', 'train-clean-360', and 'test'. It is recommened to initially download 'train-clean-100' first, which is 100 hours of LibriSpeech cleaned audio. 


## Generate CSV File with Audio Metadata and Mappings to Audio Files

The training file only works when reading metadata about the audio files from a CSV file. In order to create this CSV file, simply run the 'generate_audio_paths_excel.py' script in the 'generate_audio_mappings' directory with the following arguments:

    python3 generate_audio_paths_excel.py --path /path/to/data/dir --csv_name CSV_NAME
    
    # example
    python3 generate_audio_paths_excel.py --path /home/ubuntu/project/Data/LibriSpeech/train-clean-100 --csv_name librispeech_train_100.csv
    
1. --path: Provide the full path to the data directory where the data is stored. EX: '--path /home/ubuntu/project/Data/LibriSpeech/dev-clean'
2. --csv_name: Provide any name you would like the CSV file to be named. It is recommended you name the CSV file based on whichever split you download from the previous step with the '.csv' extension at the end. EX: if you download 'train-clean-100', name your file something like 'librispeech_train_100.csv'

After running the script, the CSV file will be saved in the same directory as the 'generate_audio_paths_excel.py'


## Pre-Train Wav2Vec2 from Scratch

Once you have successfully completed the previous two steps, you can now begin pretraining Wav2Vec2.0 from scratch using PyTorch. In order to start pretraining, you can run the 'librispeech_pytorch_from_scratch.py' script in the 'pretraining_from_scratch' directory with the following arguments:

    python3 librispeech_pytorch_from_scratch.py --batch_size BATCH_SIZE --num_epochs NUM_EPOCHS --path_to_csv FULL_PATH_TO_CSV --num_training_samples NUM_TRAINING_SAMPLES
    
    # example
    python3 librispeech_pytorch_from_scratch.py --batch_size 16 --num_epochs 50 --path_to_csv /your/path/to/csv/file --num_training_samples 10000

    
1. --batch_size: Define batch size
2. --num_epochs: Define number of epochs
3. --path_to_csv: provide full path to your CSV file generated in the previous step
4. --num_training_samples: Define number of training samples. This parameter was made in order to cut down the size of training for computation reasons. If you are using the 'train-clean-100' dataset, you can provide any number from 0 to 28530 or provide 'all' for the full training split. EX. '--num_training_samples 500' will provide only 500 audio files for training. 

