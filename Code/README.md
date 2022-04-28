#  Code Repository to Pre-Train, Fine-Tune, and Classify the Wav2Vec2.0 Model

Above are 4 different directories that make up this repository, including how to download and install LibriSpeech, pre-train Wav2Vec2.0 on LibriSpeech, fine-tune the model on TI-MIT, and finally apply Wav2Vec2.0 on a speech classification dataset. Below highlights how each directory works, depending on what you are interested in.

To make sure everything works correctly, you should do each of the below steps in order following the instructions in each directory carefully. 

## 1) Download LibriSpeech and Create Metadata File - 'generate_audio_mappings'

In this directory exists scripts that first allow you to download multiple different data splits on the LibriSpeech dataset: 'dev', 'train-clean-100', 'train-clean-360', 'test'. After downloading the data split, there exists another script to create a metadata file including all of the metadata and text information about the LibriSpeech data. 

Since the goal of this project was to develop everything from scratch, these scripts are important in order to make sure the code runs correctly. To see a full list of instructions on how to download, install, and create metadata file, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code/generate_audio_mappings).

## 2) Pre-Train Wav2Vec2.0 on LibriSpeech - 'pretraining_wav2vec'

After downloading LibriSpeech, now you can pre-train Wav2Vec2.0 on LibriSpeech. The goal here is to completely pre-train the model from scratch using custom-built methods and a custom pipeline. There are multiple options you can choose from when it comes to pre-training, so feel free to use whichever framework you're most comfortable with.

Pre-training is computationally expensive, meaning in order to successfully pre-train Wav2Vec2.0, it is recommended to have the necessary computation power (GPU's, CPU's, Memory, etc.). To see how to pre-train Wav2Vec2.0, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code/pretraining_wav2vec).

## 3) Fine-Tune Wav2Vec2.0 on TI-MIT - 'finetuning_wav2vec'

After pre-training, you now have the ability to fine-tune the pre-trained model on the TI-MIT dataset. You have multiple options for fine-tuning the model, where you can use your own pre-trained moodel, the original Wav2Vec2.0 model, or a medium or small-sized version of the Wav2Vec2.0 model, where the overall size of the model was decreased to improve training time. 

The medium and small-sized models were created by changing the architecture of Wav2Vec2.0, where we set out to see if we can decrease the number of trainable parameters and see if we can achieve similar results to the original model. To see how to fine-tune Wav2Vec2.0 on TI-MIT, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code/finetuning_wav2vec). 

## 4) Speech Classification for Wav2Vec2.0 on RAVDESS - 'speech_classification'

Now that you have successfully pre-trained Wav2Vec2.0 from scratch and fine-tuned the model, we can now use these models and apply them to a speech classification dataset where we aim to classify the emotion of different speakers from the RAVDESS dataset. 

To make Wav2Vec2.0 a classification model, we simply added on a classification head to the architecture to correctly predict the right emotion of each speaker. To see how to run your model on this speech classification dataset, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code/speech_classification).

## 5) Vocab Files for LibriSpeech & TI-MIT

During pre-training and fine-tuning, a custom vocabulary is generated for both LibriSpeech and TI-MIT, as the vocabulary is a necessary part in order to create a Wav2Vec2.0 Tokenizer, Feature Extractor, and Processor. 

This directory includes the outcome of the custom Vocab.json files, and what they should look like when pre-training and fine-tuning. To see the vocab files for LibriSpeech and TI-MIT, click [HERE](https://github.com/tristinjohnson/PreTraining-ASR-From-Scratch/tree/main/Code/vocab).
