# Apply Wav2Vec2.0 Towards a Speech Classification Dataset

After fine-tuning the model, now we can apply the fine-tuned Wav2Vec2.0 model towards a speech classification dataset, specifically RAVDESS where we aim to correctly classify the emotion of different speakers. In order to make Wav2Vec2.0 classify speech, we added a custom classification head to the end of the model in order for the output to be a classifier.  

## The Ryerson Audio-Visual Database of Emotional Speech and Songs (RAVDESS)

RAVDESS is a speech corpus that contains 24 different speakers, 12 male and 12 female, vocalizing lexically matched statements in a similar and neutral North American accent. For the speech files, the types of emotion include calm, happy, sad, angry, fearful, surprise and disgust, where the song files include the emotions of calm, happy, sad, angry, and fearful. Since we are focusing on speech classification, we only use the speech and song audio files from this dataset, and none of the video data. To learn more about this dataset, click [HERE](https://zenodo.org/record/1188976#.YmrvqfPMLAM). 

## Download and Install RAVDESS Speech and Songs dataset

Before classification, you first need to download the speech and song audio files from RAVDESS. As mentioned before, we will not be including the video files of this dataset. In order to download the data, run the 'download_and_map_emotion_data.py':

    python3 download_and_map_emotion_data.py

This script will create a directory labeled 'classification_data' with two sub-directories 'songs' and 'speech'. Inside the 'songs' directory, all of the audio files that are songs will be stored, and inside 'speech', all the audio files that are just speech will be stored.

After downloading the data, the above script will automatically create a CSV file titled 'emotional_data_mappings.csv' that includes all the metadata about each audio file, along with the emotional class. Once this CSV file has been created, now we can apply Wav2Vec2.0 to this classification problem.

## Train Wav2Vec2.0 for Speech Classification

Now that the data has been downloaded, we can train the fine-tuned Wav2Vec2.0 models on the RAVDESS emotional speech classification. To train the model, run the 'wav2vec2_classification.py' script with the following arguments:

    python3 wav2vec2_classification.py --config_path CONFIG_PATH --model_path MODEL_PATH

    # example
    python3 wav2vec2_classification.py --config_path /path/to/your/config.json --model_path /path/to/your/model.bin

1) --config_path: full path to the model configuration
2) --model_path: full path to the model

After fine-tuning the model on the TI-MIT dataset, both the configuration and the model weights would have been saved in that directory. Therefore, in order to apply either the full, medium, or small-sized Wav2Vec2.0 model, you will need to add the configuration.json file to make sure the correct model architecture is being used, along with the corresponding model weights. 

Applying the full Wav2Vec2.0 model achieved an accuracy of 89.121%, while the medium-sized model obtained a close 88.897% accuracy, and the small-sized model achieved an accuracy of 85.776%.
