"""
###############################################################
THIS IS NOT COMPLETE: NEED TO TEST MODEL THROUGHPUT
    AND MAKE OPTION TO SAVE BEST MODEL AFTER EPOCH BASED ON WER
###############################################################
Tristin Johnson
March 25th, 2022

Training Wav2Vec2 on LibriSpeech data by loading in pre-trained model from HuggingFace.
This script has uses custom PyTorch packages in order to train on the pre-trained model.
"""

# import various python packages
import pandas as pd
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from torch.utils.data import Dataset, DataLoader
from jiwer import wer
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import librosa
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


batch_size = 10  # 2
learning_rate = 0.00001
num_epochs = 20


# load in librispeech dev mappings, create full_path column
librispeech = pd.read_csv('../generate_audio_mappings/librispeech_train_mappings.csv')
librispeech = librispeech[['audio', 'audio_path', 'text_translation']]
librispeech['full_audio_path'] = librispeech['audio_path'] + librispeech['audio']

test = librispeech[0:5000]  # 1000


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


# load pretrained Wav2Vec2 Model and Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").cuda()

model_config = model.config.to_json_file('wav2vec_config.json')
#with open('wav2vec2_config.json', 'w') as config:
    #json.dump(model.config, config)

# get data collator with Padding
data_collator = DataCollatorCTCWithPadding(processor=processor)

# load raw dataset into PyTorch DataLoader
#ds_librispeech = LibriSpeechDS(librispeech)
ds_librispeech = LibriSpeechDS(test)
loader_libirspeech = DataLoader(ds_librispeech, batch_size=batch_size, shuffle=False, collate_fn=data_collator)


# define optimizer -> AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# put model in training mode
model.train()

model_best_wer = 0

# train the model
for epoch in range(1, num_epochs+1):

    total_error, total_train_steps, train_loss, total_loss = 0, 0, 0, 0

    with tqdm(total=len(loader_libirspeech), desc=f'Epoch {epoch}') as pbar:
        
        for data in loader_libirspeech:
            input_values = data['input_values'].cuda()
            target = data['labels'].cuda()

            data1 = {'input_values': input_values, 'labels': target}

            optimizer.zero_grad()
            output = model(input_values).logits
            loss = model(**data1).loss
            loss.backward()
            optimizer.step()

            _, pred_ids = torch.max(output, dim=-1)

            target_text = processor.batch_decode(target.cpu())
            pred_text = processor.batch_decode(pred_ids.cpu())

            total_train_steps += 1
            train_loss += loss.item()
            total_loss = train_loss / total_train_steps

            error = wer(target_text, pred_text)
            total_error += error
            final_wer = total_error / total_train_steps

            #print('TARGET TEXT: ', target_text[0:2])
            #print('PRED TEXT: ', pred_text[0:2])

            pbar.update(1)
            #pbar.set_postfix_str(f'B_WER:{error:0.3f} T_WER:{final_wer:0.3f} Loss: {total_loss:0.3f}')
            pbar.set_postfix_str(f'WER: {final_wer:0.3f} -> Loss: {total_loss:0.3f}')

    print(f'Epoch {epoch} final stats: Loss --> {total_loss:0.5f} WER --> {final_wer:0.5f}\n')

    model_wer = final_wer

    if model_wer > model_best_wer:
        torch.save(model.state_dict(), 'wav2vec2_best_model.pt')

        if epoch == 1:
            print('This model has been saved!\n')
        else:
            print('This model out-performed previous models and has been saved!\n')

        model_best_wer = model_wer


