"""
Tristin Johnson
May 2nd, 2022

Script to train and test the fine-tuned Wav2Vec2 model on a speech classification dataset
by adding a classification head to the model.
"""
# import various python packages
import torch
from torch import nn
from transformers import Wav2Vec2Processor, AutoConfig
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from transformers.file_utils import ModelOutput
import librosa
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import argparse
import warnings
warnings.filterwarnings('ignore')


# define training parameters
batch_size = 2
num_epochs = 50
learning_rate = 0.00001

# define number of labels for dataset
num_labels = 8
label_list = list(range(num_labels))

# put model on GPU if available, else CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device: ', device)


# function to get input values from .WAV files and the target labels
def prepare_dataset(data):
    data['input_values'], data['labels'] = "", ""

    for i in range(len(data)):
        waveform_array, sampling_rate = librosa.load(data['full_audio_path'][i], sr=16000)

        data['input_values'][i] = processor(waveform_array, sampling_rate=sampling_rate).input_values[0]
        data['input_values'][i] = torch.from_numpy(data['input_values'][i])

        with processor.as_target_processor():
            data['labels'][i] = torch.tensor(data['emotion'][i])

    return data


# Class to get input values and target labels from .CSV file into PyTorch Dataset
class EmotionalDS(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prepped_ds = prepare_dataset(self.data)

        input_data = prepped_ds.loc[index, 'input_values']

        labels = prepped_ds.loc[index, 'labels']

        return {'input_values': input_data, 'labels': labels}


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
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.int if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch['labels'] = torch.tensor(label_features, dtype=d_type)

        return batch


# dataclass to define output of Wav2Vec2 Classifier
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Define a classification head to output correct number of labels in predictions
class CustomClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input):
        x = input

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


# class to get the Wav2Vec configuration file and append the classification head
class CustomWav2VecSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        #self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = CustomClassificationHead(config)

        self.init_weights()

    # custom forward propgation function to get outputs from Wav2Vec then get classification outputs
    def forward(self, input_values,
                attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values, attention_mask=attention_mask, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return SpeechClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


# train the model
def train_and_test_wav2vec2(model, optimizer, criterion, training_ds, testing_ds):

    # send model to device and put in training mode
    model.to(device)
    model.train()

    model_best_acc = 0

    # train the model
    for epoch in range(1, num_epochs + 1):

        # training information
        train_loss, train_steps = 0, 0
        corr_pred_train, total_pred_train = 0, 0

        with tqdm(total=len(training_ds), desc=f'Training -> Epoch {epoch}') as pbar:

            for data in training_ds:
                # put target and input values on device
                input_values = data['input_values'].to(device)
                target = data['labels'].type(torch.LongTensor).to(device)

                # get model output and loss
                optimizer.zero_grad()
                output = model(input_values).logits
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1

                # make prediction
                pred = torch.argmax(output, dim=1)

                # calculate num correct
                corr_pred_train += (pred == target).sum().item()
                total_pred_train += pred.shape[0]

                # calculate accuracy and loss
                train_acc = corr_pred_train / total_pred_train
                total_train_loss = train_loss / train_steps

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {total_train_loss:0.4f}, Acc: {train_acc:0.5f}')

        # put model in evaluation model
        model.eval()

        test_loss, test_steps = 0, 0
        corr_pred_test, total_pred_test = 0, 0

        # test the model
        with torch.no_grad():
            with tqdm(total=len(testing_ds), desc=f'Testing -> Epoch {epoch}') as pbar:
                for data in testing_ds:
                    # put target and input values on device
                    input_values = data['input_values'].to(device)
                    target = data['labels'].type(torch.LongTensor).to(device)

                    optimizer.zero_grad()
                    output = model(input_values).logits
                    loss = criterion(output, target)

                    test_loss += loss.item()
                    test_steps += 1

                    # make prediction
                    pred = torch.argmax(output, dim=1)

                    # calculate num correct
                    corr_pred_test += (pred == target).sum().item()
                    total_pred_test += pred.shape[0]

                    # calculate accuracy and loss
                    test_acc = corr_pred_test / total_pred_test
                    total_test_loss = test_loss / test_steps

                    pbar.update(1)
                    pbar.set_postfix_str(f'Loss: {total_test_loss:0.4f}, Acc: {test_acc:0.5f}')

        epoch_acc = test_acc

        # if accuracy is better than previous accuracy, save the model
        if epoch_acc > model_best_acc:
            torch.save(model.state_dict(), 'best_classification_model.pt')
            print('\nYour model has been saved!\n')
            model_best_acc = epoch_acc

        else:
            print('\nThis model did not out-perform precious model accuracy!\n')

    print('\nTraining and testing is complete!!\n')


if __name__ == '__main__':

    # define arguments for script
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='facebook/wav2vec2-base-960h', help='full path to config.json file')
    parser.add_argument('--model_path', default='facebook/wav2vec2-base-960h', help='full path to model')
    args = parser.parse_args()

    # read in emotional dataset from .CSV file generated in 'generate_audio_paths.excel'
    emotion_data = pd.read_csv('emotional_data_mappings.csv')
    emotion_data = emotion_data[['audio_name', 'audio_path', 'audio_type', 'emotion']]
    emotion_data['full_audio_path'] = emotion_data['audio_path'] + emotion_data['audio_name']
    emotion_data['emotion'] = emotion_data['emotion'] - 1

    # define pooling mode: 'mean', 'max', 'min'
    pooling_mode = 'mean'

    # get config from Wav2Vec2 Base
    config = AutoConfig.from_pretrained(args.config_path,
                                        num_labels=num_labels,
                                        finetuning_task='wav2vec2_clf',
                                        label2id={label: i for i, label in enumerate(label_list)},
                                        id2label={i: label for i, label in enumerate(label_list)},
                                        problem_type='single_label_classification')

    # Define Wav2Vec2 Processor and Model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = CustomWav2VecSpeechClassification.from_pretrained(args.model_path, config=config)

    # call custom Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # define optimizer and criterion (loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()

    # get custom torch DataSet
    ds_emotion = EmotionalDS(emotion_data)

    # split into training and testing (80% training, 20% testing)
    train_len = round(len(ds_emotion) * 0.8)
    test_len = round(len(ds_emotion) * 0.2)

    train_ds, test_ds = random_split(ds_emotion, [train_len, test_len])

    # put data into torch DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=8)

    # train and test the model
    train_and_test_wav2vec2(model, optimizer, criterion, train_loader, test_loader)

