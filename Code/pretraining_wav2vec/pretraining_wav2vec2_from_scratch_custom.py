"""
Tristin Johnson
May 2nd, 2022

Pre-training Wav2Vec2.0 on LibriSpeech using custom functions and methods including:
Data Pre-processing, PyTorch DataSet/DataLoader, vocab file, data collator, PyTorch training loop
"""
# import various python packages
import pandas as pd
import re
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Config
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset as HF_dataset
from jiwer import wer
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import librosa
from tqdm import tqdm
import argparse
import json
import warnings
warnings.filterwarnings('ignore')


# load raw audio file with sampling_rate = 16kHz
def load_file(audio_file):
    waveform, sampling_rate = librosa.load(audio_file, sr=16000)

    return waveform, sampling_rate


# define list of chars to ignore in text
chars_ignore = '[\,?.!-;:\"]'


# remove special characters from text
def remove_char(data):
    for i in range(len(data)):
        data['text_translation'][i] = re.sub(chars_ignore, '', data['text_translation'][i])

    return data


# function to extract all characters from the text to create custom vocab
def extract_all_chars(batch):
    all_text = " ".join(batch['text_translation'])
    vocab = list(set(all_text))

    return {"vocab": [vocab], "all_text": [all_text]}


def create_custom_vocab(dataset):
    print('Creating custom vocab from Librispeech! ...')
    librispeech_vocab = HF_dataset.from_pandas(dataset)

    # extract all characters from LibriSpeech and create vocab list
    vocabs = librispeech_vocab.map(extract_all_chars, batched=True, batch_size=1, keep_in_memory=True)
    all_vocab = vocabs['vocab']
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

    print('Custom Vocab created: ', vocab_dict)
    print('Length of vocab file (should be 32): ', len(vocab_dict))

    with open('../vocab/vocab_librispeech.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    print('Custom vocab has been created and saved in the \'vocab\' directory as \'vocab_librispeech.json\'')


# function to extract waveform from audio, tokenize audio, and get input values for model
def prepare_dataset(data):
    data['input_values'], data['labels'] = "", ""
    for i in range(len(data)):
        waveform_array, sampling_rate = librosa.load(data['full_audio_path'][i], sr=16000)

        data['input_values'][i] = processor(waveform_array, sampling_rate=sampling_rate).input_values[0]
        data['input_values'][i] = torch.from_numpy(data['input_values'][i])

        with processor.as_target_processor():
            data['labels'][i] = processor.tokenizer(data['text_translation'][i]).input_ids

    return data


# custom LibriSpeech PyTorch Dataset
class LibriSpeechDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # clean the data
        cleaned_ds = remove_char(self.data)

        # prep the data: get input_ids and text labels
        prepped_ds = prepare_dataset(cleaned_ds)

        # get input data (audio) # maybe get rid of values
        input_data = prepped_ds.loc[index, 'input_values']

        # get tokenized text (target)
        tokenized_txt = prepped_ds.loc[index, 'labels']

        return {'input_values': input_data, 'labels': tokenized_txt}


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
            padding=True,
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


def train_and_test_wav2vec_from_scratch(librispeech_train, librispeech_test, num_epochs, model, optimizer, processor):
    # put model in training mode
    model.train()

    model_best_wer = 0

    # train the model
    for epoch in range(1, num_epochs + 1):

        train_batch_wer, total_train_steps, train_loss = 0, 0, 0

        with tqdm(total=len(librispeech_train), desc=f'Epoch {epoch}') as pbar:

            for data in librispeech_train:
                # send input_values and target labels to GPU
                input_values = data['input_values'].cuda()
                target = data['labels'].cuda()

                data1 = {'input_values': input_values, 'labels': target}

                optimizer.zero_grad()
                output = model(input_values).logits
                loss = model(**data1).loss
                loss.backward()
                optimizer.step()

                # get prediction ids
                pred_ids = torch.argmax(output, dim=-1)

                # decode target and prediction
                target_text = processor.batch_decode(target.cpu())
                pred_text = processor.batch_decode(pred_ids.cpu())

                # calculate current batch WER
                batch_error = wer(target_text, pred_text)
                total_train_steps += 1

                # calculate overall WER
                train_batch_wer += batch_error
                train_wer = train_batch_wer / total_train_steps
                train_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {train_loss / total_train_steps:0.3f}, WER: {train_wer:0.3f}')

        # put model in evaluation mode
        model.eval()

        test_batch_wer, total_test_steps, test_loss = 0, 0, 0

        # test the model on test split
        with torch.no_grad():
            with tqdm(total=len(librispeech_test), desc=f'Epoch {epoch}') as pbar:

                for data in librispeech_test:
                    input_values = data['input_values'].cuda()
                    target = data['labels'].cuda()

                    data1 = {'input_values': input_values, 'labels': target}

                    optimizer.zero_grad()
                    output = model(input_values).logits
                    loss = model(**data1).loss

                    # get prediction ids
                    pred_ids = torch.argmax(output, dim=-1)

                    # decode target and prediction
                    target_text = processor.batch_decode(target.cpu())
                    pred_text = processor.batch_decode(pred_ids.cpu())

                    # calculate current batch WER
                    batch_error = wer(target_text, pred_text)
                    total_test_steps += 1

                    # calculate overall WER
                    test_batch_wer += batch_error
                    test_wer = train_batch_wer / total_train_steps
                    test_loss += loss.item()

                    pbar.update(1)
                    pbar.set_postfix_str(f'Loss: {test_loss / total_test_steps:0.3f}, WER: {test_wer:0.3f}')

        epoch_wer = test_wer

        # if WER is greater than previous WER, save the model
        if epoch_wer > model_best_wer:

            torch.save(model.state_dict(), 'best_wav2vec2_model.pt')
            print('Your model has been saved!!\n')
            model_best_wer = epoch_wer

        else:
            print('This model did not out-perform previous WER!\n')

    print('Training and Testing is complete!')


# main
if __name__ == '__main__':

    # define arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, help='batch size')
    parser.add_argument('--num_epochs', default=50, help='number of epochs for training')
    parser.add_argument('--path_to_csv', default='../generate_audio_mappings/librispeech_train_mappings.csv', help='full path to your CSV file')
    parser.add_argument('--num_training_samples', default=20000, help='number of samples for training: either any number from 0-28530 or all (all is full LibriSpeech training)')
    args = parser.parse_args()

    # load in librispeech dev mappings, create full_path column
    librispeech = pd.read_csv(args.path_to_csv)
    librispeech = librispeech[['audio', 'audio_path', 'text_translation']]
    librispeech['full_audio_path'] = librispeech['audio_path'] + librispeech['audio']

    # create custom vocab from LibriSpeech data
    create_custom_vocab(librispeech)

    # load Wav2Vec2 tokenizer with custom vocab, create feature extractor, implement processor
    tokenizer = Wav2Vec2CTCTokenizer('../vocab/vocab_librispeech.json', unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # load Wav2Vec2 Config and Model
    config = Wav2Vec2Config()
    model = Wav2Vec2ForCTC(config).cuda()
    model.config.architectures = ["Wav2Vec2ForCTC"]

    # get data collator with Padding
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # define number of training samples and get custom torch DataSet
    if args.num_training_samples == 'all':
        librispeech_ds = LibriSpeechDS(librispeech)
    else:
        librispeech_ds = LibriSpeechDS(librispeech[0: int(args.num_training_samples)])

    # split into training and testing
    train_len = round(len(librispeech_ds) * 0.8)
    test_len = round(len(librispeech_ds) * 0.2)

    train_ds, test_ds = random_split(librispeech_ds, [train_len, test_len])

    # load dataset into PyTorch DataLoader
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=data_collator)

    # define optimizer -> AdamW
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # train Wav2Vec2 on LibriSpeech
    print(f'\nStarting training from scratch using Wav2Vec2 on Librispeech with batch_size: {args.batch_size} '
          f'-> num_epochs: {args.num_epochs} -> num_samples: {args.num_training_samples}\n')

    # train and test the model
    train_and_test_wav2vec_from_scratch(train_loader, test_loader, int(args.num_epochs), model, optimizer, processor)
