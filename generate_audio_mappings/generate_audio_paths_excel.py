"""
###############################################################
THIS IS COMPLETE: REMOVE THIS LINE WHEN UPLOADING TO GITHUB
###############################################################
Tristin Johnson
March 25th, 2022

Script to generate an excel file with audio metadata (audio name, id, path, text)
in order to read in audio mappings for customization.
"""
# import various python packages
import pandas as pd
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')


# function to create df and excel file including audio file name, audio path, and text translation per audio
def create_excel_audio_mappings(data_directory, data_split_name):
    # create empty df
    df = pd.DataFrame(columns=['audio', 'audio_path', 'text_translation']).transpose()

    index = 0

    # get path of all audio files
    test = glob(data_directory + '/*/*/')
    for files in test:
        files_in_dir = os.listdir(files)

        for file in files_in_dir:

            # find .txt files in each sub directory to locate corresponding audio file with text translation
            if 'txt' in file:

                # get txt path and txt length
                txt_path = open(files + file)
                txt_len = len(txt_path.readlines())
                txt_path.close()

                # open the txt file to view text translations
                with open(files + file) as f:

                    # get audio name, text translation, and path of audio file
                    for i in range(txt_len):
                        audio_name = f.readline().split()
                        text_trans = ' '.join(audio_name[1:])
                        audio_path = files

                        # all audio name, path, and text to dataframe
                        df = df.append({'audio_id': int(index), 'audio': audio_name[0] + '.flac', 'audio_path': audio_path, 'text_translation': text_trans}, ignore_index=True)
                        index += 1

    # drop NaN values, and convert text to lowercase
    df = df.dropna()
    #df['text_translation'] = df['text_translation'].str.lower()
    df['audio_id'] = df['audio_id'].astype(int)

    # export df as excel file
    df.to_csv(data_split_name)

    return df


#dev_data_directory = '/home/ubuntu/capstone/data/LibriSpeech/dev-clean'
dev_data_directory = '/home/ubuntu/capstone/data/LibriSpeech/train-clean-100'
dev_data = create_excel_audio_mappings(dev_data_directory, 'librispeech_train_mappings.csv')
print(dev_data)
print(dev_data.iloc[-1, 0])
print(dev_data.iloc[-1, 1])
print(dev_data.iloc[-1, 2])
print(type(dev_data.iloc[-1, 0]))
print(len(dev_data))

dev_data['audio_id'] = dev_data['audio_id'].astype(int)
print(type(dev_data.iloc[-1, 0]))
