# All Custom-Made Vocab Files for Wav2Vec2.0 Tokenizer, Feature Extractor & Processor

One of the core components to creating an accurate model is to define the tokenizer. Previously, if you pre-trained or fine-tuned the model using other directories, you will have also generated your own vocabulary for the Wav2Vec2.0 tokenizer. 

If you pre-trained the model using the pre-training scripts, a JSON file titled 'vocab_librispeech.json' would have been generated automatically as we used LibriSpeech for pre-training, similar to the one here. The same applies for fine-tuning, a custom vocabulary file would have generated titled 'vocab_timit.json' as we fine-tuned the model on the TI-MIT dataset, similarly to the one here. 

To see how to create a custom vocabulary, refer to either the fine-tuning directory or the pre-training directory, and look in the scripts for a function called 'create_custom_vocab'.