from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

"""processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')


def map_to_array(batch):
    speech, _ = sf.read(batch['file'])
    batch['speech'] = speech

    return batch


ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(ds['speech'][:2], return_tensors='pt', padding='longest').input_values
logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
transcriptions = processor.batch_decode(pred_ids)

print(transcriptions)"""

librispeech_eval = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')


def map_to_array(batch):
    speech, _ = sf.read(batch['file'])
    batch['speech'] = speech

    return batch


librispeech_eval = librispeech_eval.map(map_to_array)


def map_to_pred(batch):
    input_values = processor(batch["speech"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cpu")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch


result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

for i in range(len(result)):
    print('TEXT: ', result['text'][i])
    print('TRANSCRIPTION: ', result['transcription'][i])
    print()

