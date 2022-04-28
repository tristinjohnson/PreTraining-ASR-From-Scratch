# Fine-Tuning Wav2Vec2.0 for Speech Recognition

This directory allows you to fine-tune the Wav2Vec2.0 model on the TI-MIT dataset. Furthermore, you have the option to use the original and full Wav2Vec2.0 model, or you can use a medium or small-sized model that I created in order to shrink the size of the model with the idea to decrease training time. 

## TI-MIT Speech Corpus

TI-MIT is an acoustic-phonetic speech corpus to further help with the development of ASR. This dataet includes 360 speakers of 8 different dialects of American English, each reading 10 phonetically rich sentences. The unique part of TI-MIT is the fact that the dataset also comes with the phonemes of each speech recording, meaning a model can be trained using just the phonemes of speech. However, for our purposes, we will be sticking with the textual data to enhance speech recognition. To learn more about TI-MIT, click [HERE](https://catalog.ldc.upenn.edu/LDC93s1).

## Fine-Tune Wav2Vec2.0 on TI-MIT

The script is simple as all you need to define is the type of model you would like to train. Simply run the 'training_wav2vec2.py' script using the following argument:

    python3 training_wav2vec2.py --model_type MODEL_TYPE

    # example
    python3 training_wav2vec2.py --model_type full

1) --model_type: which type of model you would like ot train: full, medium, small

One of the contributions to the Wav2Vec2.0 model is to cut down the number of trainable parameters and see if we can obtain an accurate model but with much less training time. The full Wav2Vec2.0 model has about 95 million trainable parameters, and we were able to achieve a WER of 16.298%. By decreasing some of the model parameters, we created a medium-sized model with around 61 trainable parameters obtaining a WER of 31.578%, and a small-sized model with around 36 million trainable obtaining a WER of 53.434%.

Feel free to customize the model parameters (batch size, epochs, etc.) within the script. Furthermore, this script uses HiggingFace's Trainer and TrainingArgs, so every best iteration of the model will be saved as a checkpoint, and you can change the number of save limits along with how often to perform an evaluation.