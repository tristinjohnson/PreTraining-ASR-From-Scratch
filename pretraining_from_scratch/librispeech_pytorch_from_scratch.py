import random
import pandas as pd
import numpy as np
import re
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Config, Wav2Vec2Model
import torch
from torch.utils.data import Dataset, DataLoader
from jiwer import wer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import librosa
from datasets import load_metric
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# load in librispeech dev mappings, create full_path column
librispeech = pd.read_csv('speech_paths/librispeech_train_mappings.csv')
librispeech = librispeech[['audio', 'audio_path', 'text_translation']]
librispeech['full_audio_path'] = librispeech['audio_path'] + librispeech['audio']

#test = librispeech[0:40]
test = librispeech[0:15000]


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


# load Wav2Vec2 tokenizer with custom vocab, create feature extractor, implement processor
tokenizer = Wav2Vec2CTCTokenizer('vocab/vocab_librispeech.json', unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
config = Wav2Vec2Config()
model = Wav2Vec2ForCTC(config)


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


#librispeech_pre = prepare_dataset(librispeech)


# custom LibriSpeech PyTorch Dataset
class LibriSpeechDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get files
        #files = self.data['audio'][index]

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


# get data collator with Padding
data_collator = DataCollatorCTCWithPadding(processor=processor)

# load raw dataset into PyTorch DataLoader
#ds_librispeech = LibriSpeechDS(librispeech)
ds_librispeech = LibriSpeechDS(test)
loader_libirspeech = DataLoader(ds_librispeech, batch_size=4, shuffle=False, collate_fn=data_collator)

# load Wav2Vec2 config file and model
#config = Wav2Vec2Config()
#model = Wav2Vec2ForCTC(config)
#processor_1 = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base')
#model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')


# define optimizer -> AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)

# put model in training mode
model.train()

# train the model
for epoch in range(60):

    total_error, total_train_steps = 0, 0

    with tqdm(total=len(loader_libirspeech), desc=f'Epoch {epoch}') as pbar:
        
        for data in loader_libirspeech:
            input_values = data['input_values']
            target = data['labels']

            optimizer.zero_grad()
            output = model(input_values).logits
            loss = model(**data).loss
            loss.backward()
            optimizer.step()

            _, pred_ids = torch.max(output, dim=-1)

            target_text = processor.batch_decode(target)
            pred_text = processor.batch_decode(pred_ids)
            #print('\n', pred_text, '\n')

            #print('Target Text Ex: ', target_text[0])
            #print('Pred Text Ex: ', pred_text[0])

            error = wer(target_text, pred_text)
            total_train_steps += 1
            total_error += error
            final_wer = total_error / total_train_steps

            if total_train_steps % 10 == 0:
                print(f'\n\nPred checkpoint at epoch: {epoch} -> steps: {total_train_steps} ---> \n{pred_text}')

            pbar.update(1)
            pbar.set_postfix_str(f'B_WER: {error:0.3f} -> T_WER: {final_wer:0.3f}')

    if epoch % 5 == 0:
        print(f'\n\n\nPred Text of last batch at epoch {epoch} --> {pred_text}')

        torch.save(model.state_dict(), f'dev_wav2vec_model_{epoch}.pt')

