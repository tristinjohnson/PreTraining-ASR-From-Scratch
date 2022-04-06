# import various python packages
import pandas as pd
import re
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Config
import torch
from torch.utils.data import Dataset, DataLoader
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



"""# load in librispeech dev mappings, create full_path column
librispeech = pd.read_csv('../generate_audio_mappings/librispeech_train_mappings.csv')
librispeech = librispeech[['audio', 'audio_path', 'text_translation']]
librispeech['full_audio_path'] = librispeech['audio_path'] + librispeech['audio']

test = librispeech[0:200]"""


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

    vocabs = librispeech_vocab.map(extract_all_chars, batched=True, batch_size=1, keep_in_memory=True)
    vocabs_list = list(set(vocabs['vocab'][0]))

    vocab_dict = {v: k for k, v in enumerate(vocabs_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # doesnt include J, Q, X, Z
    vocab_dict["J"] = len(vocab_dict)
    vocab_dict["Q"] = len(vocab_dict)
    vocab_dict["X"] = len(vocab_dict)
    vocab_dict["Z"] = len(vocab_dict)
    vocab_dict["<unk>"] = len(vocab_dict)
    vocab_dict["<pad>"] = len(vocab_dict)
    vocab_dict["<s>"] = len(vocab_dict)
    vocab_dict["</s>"] = len(vocab_dict)
    vocab_dict["'"] = len(vocab_dict)

    print('Custom Vocab created: ', vocab_dict)
    print('Length of vocab file (should be 32): ', len(vocab_dict))

    with open('./custom_librispeech_vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    print('Custom vocab has been created and saved in this directory as \'custom_librispeech_vocab.json\'')


#create_custom_vocab(librispeech)



"""# load Wav2Vec2 tokenizer with custom vocab, create feature extractor, implement processor
tokenizer = Wav2Vec2CTCTokenizer('../vocab/custom_librispeech_vocab.json', unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# load Wav2Vec2 Config and Model
#config = Wav2Vec2Config().from_pretrained('../pretraining_from_model/wav2vec_config.json')
config = Wav2Vec2Config()
model = Wav2Vec2ForCTC(config).cuda()

model.config.architectures = ["Wav2Vec2ForCTC"]"""


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


"""# get data collator with Padding
data_collator = DataCollatorCTCWithPadding(processor=processor)

# load raw dataset into PyTorch DataLoader
#ds_librispeech = LibriSpeechDS(librispeech)
ds_librispeech = LibriSpeechDS(test)
loader_libirspeech = DataLoader(ds_librispeech, batch_size=batch_size, shuffle=False, collate_fn=data_collator)


# define optimizer -> AdamW
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ctc_loss = torch.nn.CTCLoss()"""


def train_wav2vec_from_scratch(librispeech_ds, num_epochs, model, optimizer, processor):
    # put model in training mode
    model.train()

    model_best_wer = 0

    # train the model
    for epoch in range(1, num_epochs + 1):

        batch_wer, total_train_steps = 0, 0

        with tqdm(total=len(librispeech_ds), desc=f'Epoch {epoch}') as pbar:

            for data in librispeech_ds:
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

                #print('\nTARGET TEXT: ', target_text, '\nPREDICTED TEXT: ', pred_text, '\n')
                #print('Target Text Ex: ', target_text[0])
                #print('Pred Text Ex: ', pred_text[0])

                # calculate current batch WER
                batch_error = wer(target_text, pred_text)
                total_train_steps += 1

                # calculate overall WER
                batch_wer += batch_error
                total_wer = batch_wer / total_train_steps

                if total_train_steps % 10 == 0:
                    print(f'\n\nPrediction checkpoint at epoch: {epoch} and steps: {total_train_steps} ---> \n{pred_text}')

                pbar.update(1)
                pbar.set_postfix_str(f'Batch_WER: {batch_error:0.3f} -> Total_WER: {total_wer:0.3f}')

        current_epoch_wer = total_wer

        # if WER is greater than previous WER, save the model
        if current_epoch_wer > model_best_wer:

            torch.save(model.state_dict(), 'best_wav2vec2_model.pt')
            print('Your model has been saved!!')


# main
if __name__ == '__main__':

    # define arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=2, help='batch size')
    parser.add_argument('--num_epochs', default=50, help='number of epochs for training')
    parser.add_argument('--path_to_csv', required=True, help='full path to your CSV file')
    parser.add_argument('--num_training_samples', required=True, help='number of samples for training: either any number from 0-23580 or all (all is full LibriSpeech training)')
    args = parser.parse_args()

    # load in librispeech dev mappings, create full_path column
    #librispeech = pd.read_csv('../generate_audio_mappings/librispeech_train_mappings.csv')
    librispeech = pd.read_csv(args.path_to_csv)
    librispeech = librispeech[['audio', 'audio_path', 'text_translation']]
    librispeech['full_audio_path'] = librispeech['audio_path'] + librispeech['audio']

    # create custom vocab from LibriSpeech data
    create_custom_vocab(librispeech)

    # load Wav2Vec2 tokenizer with custom vocab, create feature extractor, implement processor
    tokenizer = Wav2Vec2CTCTokenizer('./custom_librispeech_vocab.json', unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # load Wav2Vec2 Config and Model
    config = Wav2Vec2Config()
    model = Wav2Vec2ForCTC(config).cuda()
    model.config.architectures = ["Wav2Vec2ForCTC"]

    # get data collator with Padding
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # define number of training samples
    if args.num_training_samples == 'all':
        ds_librispeech = LibriSpeechDS(librispeech)
    else:
        ds_librispeech = LibriSpeechDS(librispeech[0: int(args.num_training_samples)])

    # load dataset into PyTorch DataLoader
    loader_libirspeech = DataLoader(ds_librispeech, batch_size=int(args.batch_size), shuffle=False, collate_fn=data_collator)

    # define optimizer -> AdamW
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # train Wav2Vec2 on LibriSpeech
    print(f'\nStarting training from scratch using Wav2Vec2 on Librispeech with batch_size: {args.batch_size} '
          f'-> num_epochs: {args.num_epochs} -> num_samples: {args.num_training_samples}\n')

    train_wav2vec_from_scratch(loader_libirspeech, int(args.num_epochs), model, optimizer, processor)
