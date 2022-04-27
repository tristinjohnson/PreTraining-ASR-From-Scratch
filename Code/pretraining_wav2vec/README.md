# Pre-Train Wav2Vec2.0 from Scratch

Below will explain how to run the scripts in order to pre-train Wav2Vec2.0 from scratch. It is import that you have correctly downloaded and installed LibriSpeech and created the metadata file in the previous section in order for the code to run correctly. If you haven't completed this step yet, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code/generate_audio_mappings).

You can pre-train the model in two different ways:
 1) Using custom PyTorch methods and functions
 2) Using custon HuggingFace methods and functions

Both of these methods include customized methods that were built from scratch, such as the data pre-processing, a custom Wav2Vec2 tokenizer, feature extractor, process, and a data collator function. Follow the steps below to see which method you prefer. You can also try both and compare the two different methods.

## Pre-Training using custom PyTorch Methods and Functions

The first method is using primarily PyTorch methods, with custom functionality added. Some of these custom functions include a custom PyTorch DataSet and DataLoader, along with the PyTorch training loop. In order to use this method, run the 'pretraining_wav2vec2_from_scratch_custom.py' file with the following arguments:

    python3 pretraining_wav2vec2_from_scratch_custom.py --batch_size BATCH_SIZE --num_epochs NUM_EPOCHS --path_to_csv PATH_TO_CSV --num_training_samples NUM_TRAINING_SAMPLES

    # example 
    python3 pretraining_wav2vec2_from_scratch_custom.py --batch_size 32 --num_epochs 75 --path_to_csv /path/to/metadata/file.csv --num_training_samples 10000

1. --batch_size: Define batch size
2. --num_epochs: Define number of epochs
3. --path_to_csv: provide full path to your CSV file generated in the previous step
4. --num_training_samples: Define number of training samples. 
   
The 'num_training_samples' argument was made in order to cut down the size of training for computation reasons. If you are using the 'train-clean-100' dataset, you can provide any number from 0 to 28530 or provide 'all' for the full training split. EX. '--num_training_samples 500' will provide only 500 audio files for training. The reason for this is because pre-training is very expensive computationally, as the original creators of Wav2Vec2.0 used 128 GPU's which trained for over 120 hours straight. Therefore, if you don't have the computation power, you can shrink the number of training samples that correlates with the amount of computation you have access to so that pre-training doesn't take days or weeks.

## Pre-Training using custom HuggingFace Methods and Functions

The second method is using primarily HuggingFace's methods, also with some custom functionality added. Since HuggingFace hosts the Wav2Vec2.0 model, using their Trainer and TrainingArgs packages can be effecient when pre-training as you have the ability to save more models with checkpoints and save all information from each epcoh. In order to use this method, run the 'pretraining_wav2vec2_from_scratch.py' using the following arguments:

    python3 pretraining_wav2vec2_from_scratch.py --path_to_csv PATH_TO_CSV --num_training_samples NUM_TRAINING_SAMPLES

    # example
    python3 pretraining_wav2vec2_from_scratch.py --path_to_csv path/to/metadata/file.csv --num_training_samples 15000

1) --path_to_csv: provide full path to your CSV file generated in the previous step
2) --num_training_samples: Define number of training samples

The 'num_training_samples' argument exists for the same reason as above. You can still pre-train on full LibriSpeech, however if you lack computation resources, it may be in your interest to shrink the size of the training set. Furthermore, to better customize training to your needs, you can change the values in the 'hugging_face' function inside of the script (ex. batch size, number of epochs, total save limits, save steps, evaluation steps, etc.).

