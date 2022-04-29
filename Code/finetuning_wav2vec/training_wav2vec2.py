"""
Tristin Johnson
May 2nd, 2022

Training Wav2Vec2 on TI-MIT data by loading in pre-trained model from HuggingFace.
This script uses the built-in HuggingFace packages in order to train on the pre-trained model.
"""
# import various python packages
import numpy as np
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, AutoConfig
from transformers import Trainer, TrainingArguments
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import librosa
from datasets import load_metric, load_dataset
import json
import argparse
import warnings
warnings.filterwarnings('ignore')


# define training variables
batch_size = 8
num_epochs = 60
learning_rate = 0.00001
sr = 16000


# load in the audio file with sampling_rate = 16 kHz
def load_file(audio_file):
    waveform, sampling_rate = librosa.load(audio_file, sr=sr)

    return waveform, sampling_rate


# define list of chars to ignore in text
chars_ignore = '[\,?.!-;:\"]'


# function to remove all special characters
def remove_chars(batch):
    batch['text'] = re.sub(chars_ignore, '', batch['text']).lower()

    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch['text'])
    vocab = list(set(all_text))

    return {"vocab": [vocab], "all_text": [all_text]}


def create_custom_vocab(dataset):
    print('Creating custom vocab from TIMIT! ...')

    # extract all characters from TIMIT and create vocab list
    vocabs = dataset.map(extract_all_chars, batched=True, batch_size=1, keep_in_memory=True)
    all_vocab = vocabs['train']['vocab']
    vocab_list, temp = [], set()

    for letters in all_vocab:
        for char in letters:
            if not char in temp:
                temp.add(char)
                vocab_list.append(char)

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # add missing tokens for Wav2Vec tokenizer
    vocab_dict["<unk>"] = len(vocab_dict)
    vocab_dict["<pad>"] = len(vocab_dict)
    vocab_dict["<s>"] = len(vocab_dict)
    vocab_dict["</s>"] = len(vocab_dict)
    vocab_dict["'"] = len(vocab_dict)

    print('Custom vocab created: ', vocab_dict)
    print('Length of vocab file: ', len(vocab_dict))

    # save the vocab file
    with open('../vocab/vocab_timit.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    print('Custom vocab has been created and saved in the \'vocab\' directory as \'vocab_timit.json\'')


# function to extract waveform from audio, tokenize audio, and get input values for model
def prepare_dataset(batch):

    # if 'audio' exists for TI-MIT data
    try:
        audio = batch["audio"]

        # get input values
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    # else manually get input values
    except Exception:
        wav, sr = librosa.load(batch['file'])

        # get input values
        batch["input_values"] = processor(wav, sampling_rate=16000).input_values[0]

    # get target values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids

    return batch


# custom class for Data Collator with Padding for Wav2Vec2 Data Collator
@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# load WER (Word Error Rate --> metric used for speech-to-text models)
wer_metric = load_metric('wer')


# function to get predictions and compute wer against true labels
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {'wer': wer}


# train the model using HuggingFace Trainer
def huggingface_trainer(model, data_collator, processor, librispeech_full_ds, output_dir_name):
    # implement training arguments for training
    training_args = TrainingArguments(
        output_dir=output_dir_name,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        evaluation_strategy='steps',
        num_train_epochs=num_epochs,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        warmup_steps=1000,
        save_total_limit=5,
        load_best_model_at_end=True
    )

    # implement trainer with model, datasets, metrics, training args, data collator
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=librispeech_full_ds['train'],
        eval_dataset=librispeech_full_ds['test'],
        tokenizer=processor.feature_extractor
    )

    # train the model
    trainer.train()

    # save best model at the end
    trainer.save_model(output_dir=output_dir_name)


if __name__ == '__main__':

    # define arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='full', help='Which model to use -> any of [\'full\', \'medium\', \'small\']')
    args = parser.parse_args()

    # load TIMIT dataset from HuggingFace
    timit = load_dataset('timit_asr')
    timit = timit.remove_columns(['phonetic_detail', 'word_detail', 'dialect_region', 'sentence_type', 'speaker_id', 'id'])

    # remove special chars from data
    timit = timit.map(remove_chars)
    create_custom_vocab(timit)

    #
    tokenizer = Wav2Vec2CTCTokenizer('../vocab/vocab_timit.json', unk_token='<unk>', pad_token='<pad>', word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    output_dir_name = 'timit_wav2vec2'

    # define Wav2Vec2.0 configuration
    if args.model_type == 'full':
        config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
        output_dir_name = 'timit_full_wav2vec2'

    elif args.model_type == 'medium':
        config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
        setattr(config, 'num_hidden_layers', 8)
        output_dir_name = 'timit_medium_wav2vec2'

    elif args.model_type == 'small':
        config = AutoConfig.from_pretrained('facebook/wav2vec2-base-960h')
        setattr(config, 'num_hidden_layers', 8)
        setattr(config, 'num_adaptive_layers', 3)
        setattr(config, 'num_attention_heads', 8)
        output_dir_name = 'timit_small_wav2vec2'

    # define pretrained Wav2Vec2.0 processor and model
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", config=config)

    # freeze the feature extractor
    model.freeze_feature_extractor()

    # get input values and target labels in data
    timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=8)

    # define custom Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # train and validate the model
    huggingface_trainer(model, data_collator, processor, timit, output_dir_name)

