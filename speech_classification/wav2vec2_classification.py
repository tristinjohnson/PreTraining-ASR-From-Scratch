import torch
from torch import nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

custom_path = '/home/ubuntu/capstone/wav2vec2/pretraining_from_model/librispeech_pretrained_models/checkpoint-800/'
t = '/home/ubuntu/capstone/wav2vec2/pretraining_from_model/wav2vec2_best_model.pt'

model = Wav2Vec2ForSequenceClassification.from_pretrained(custom_path)
#model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
#model.load_state_dict(torch.load('/home/ubuntu/capstone/wav2vec2/pretraining_from_model/librispeech_pretrained_models/checkpoint-800/pytorch_model.bin'))

#print(model)
print(model.config)
