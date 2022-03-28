import random
import pandas as pd
import numpy as np
import re
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset as TorchDataSet
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import librosa
from datasets import Dataset, load_metric


# load in librispeech dev mappings, create full_path column
librispeech = pd.read_csv('speech_paths/librispeech_dev_mappings.csv')
librispeech = librispeech[['audio', 'audio_path', 'text_translation']]
librispeech['full_audio_path'] = librispeech['audio_path'] + librispeech['audio']
#librispeech = librispeech.drop(librispeech.index[len(librispeech)-1])

# transform dataset to a HuggingFace Dataset
librispeech_ds = Dataset.from_pandas(librispeech)
print(librispeech_ds)


# load in the audio file with sampling_rate = 16 kHz
def load_file(audio_file):
    waveform, sampling_rate = librosa.load(audio_file, sr=16000)

    return waveform, sampling_rate


"""class LibriSpeechDS(TorchDataSet):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio_path = self.data['full_audio_path'][index]
        audio_name = self.data['audio'][index]
        flac_array, sampling_rate = load_file(audio_path)
        text = self.data['text_translation'][index]

        audio_dict = {'array': flac_array, 'path': audio_path, 'sampling_rate': sampling_rate}

        return audio_name, audio_dict, text"""


# define list of chars to ignore in text
chars_ignore = '[\,?.!-;:\"]'


# function to remove all special characters
def remove_chars(batch):
    batch['text_translation'] = re.sub(chars_ignore, '', str(batch['text_translation']))  # .lower()

    return batch


# remove special characters from librispeech
librispeech_ds = librispeech_ds.map(remove_chars)


# function to extract all characters from the text to create custom vocab
def extract_all_chars(batch):
    all_text = " ".join(batch['text_translation'])
    vocab = list(set(all_text))

    return {"vocab": [vocab], "all_text": [all_text]}


# extract all characters from librispeech
vocabs = librispeech_ds.map(extract_all_chars, batched=True, batch_size=1, keep_in_memory=True)
vocab_list = list(set(vocabs['vocab'][0]))

# create custom vocab dictionary from librispeech
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print(vocab_dict)

# replace 'space' char with '|'
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# add [UNK] and [PAD] to vocab (needed for Wav2Vec2)
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict['[PAD]'] = len(vocab_dict)
print(vocab_dict)
print(len(vocab_dict))

# save the vocab file
with open('librispeech_vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# load Wav2Vec2 tokenizer with custom vocab, create feature extractor, implement processor (combination of tokenizer and feature extractor)
tokenizer = Wav2Vec2CTCTokenizer('vocab/vocab_librispeech.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# function to extract waveform from audio, tokenize audio, and get input values for model
def prepare_dataset(batch):
    waveform_array, sampling_rate = librosa.load(batch['full_audio_path'], sr=16000)

    batch['input_values'] = processor(waveform_array, sampling_rate=sampling_rate).input_values[0]

    with processor.as_target_processor():
        batch['labels'] = processor(batch['text_translation']).input_ids

    return batch


# get input values for audio in librispeech
librispeech_ds = librispeech_ds.map(prepare_dataset)

# split dataset into 75% training and 25% testing
librispeech_train_test = librispeech_ds.train_test_split(test_size=0.2)

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


# get Wav2Vec2 Configuration and input into the model
config = Wav2Vec2Config()
model = Wav2Vec2ForCTC(config)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# use this for number of models to save:
# total steps = (num_training_samples / batch_size) * num_epochs

# implement training arguments for training
training_args = TrainingArguments(
    output_dir='librispeech_models',
    group_by_length=True,
    per_device_train_batch_size=4,
    evaluation_strategy='steps',
    num_train_epochs=2,
    gradient_checkpointing=True,
    save_steps=450,
    eval_steps=450,
    logging_steps=450,
    warmup_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True
)

# implement trainer with model, datasets, metrics, training args, data collator
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=librispeech_train_test['train'],
    eval_dataset=librispeech_train_test['test'],
    tokenizer=processor.feature_extractor
)

# train the model
trainer.train()

# save best model at the end
trainer.save_model(output_dir='librispeech_best_model')
