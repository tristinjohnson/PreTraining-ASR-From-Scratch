"""
Tristin Johnson
May 2nd, 2022

Script to download the RAVDESS Audio Speech and Songs dataset and to
generate a .CSV file with metadata about each audio file for custom training methods.
"""
# import various python packages
import os
from glob import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data_directory = os.getcwd() + '/classification_data'

# create directory to store songs from RAVDESS
os.makedirs('classification_data/songs/')
os.chdir('classification_data/songs/')

# download audio songs link and unzip:
print('Downloading and Decompressing Audio Songs from RAVDESS Emotional Classification')
os.system("wget -c https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip")
os.system("unzip Audio_Song_Actors_01-24.zip")
os.system("rm Audio_Song_Actors_01-24.zip")

# create directory to store speech from RAVDESS
os.chdir('../')
os.makedirs('speech/')
os.chdir('speech/')

# download audio speech link and unzip:
print('Downloading and Decompressing Audio Speech from RAVDESS Emotional Classification')
os.system("wget -c wget -c https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip")
os.system("unzip Audio_Speech_Actors_01-24.zip")
os.system("rm Audio_Speech_Actors_01-24.zip")


# function to create df and excel file including audio file name, audio path, and text translation per audio
def create_csv_audio_mappings(data_directory, data_split_name):
    # create empty df
    df = pd.DataFrame(columns=['audio_id', 'audio_name', 'audio_path', 'audio_type', 'emotion']).transpose()

    index = 0

    # get path of all audio files
    test = glob(data_directory + '/*/*/')
    for files in test:
        files_in_dir = os.listdir(files)

        for file in files_in_dir:
            if 'speech' in files:
                audio_type = 'speech'

            else:
                audio_type = 'song'

            audio_id = index
            audio_name = file
            audio_path = files
            emotion = file[7:8]

            # add metadata attributes to dataframe
            df = df.append({'audio_id': audio_id, 'audio_name': audio_name, 'audio_path': audio_path, 'audio_type': audio_type, 'emotion': emotion}, ignore_index=True)

            index += 1

    # drop NaN values and set audio_id as index
    df = df.dropna()
    df['audio_id'] = df['audio_id'].astype(int)
    df.set_index = df['audio_id']

    print(df.head())
    print('Length of Dataframe: ', len(df))
    print('DataFrame columns: ', df.columns)

    # export df to .CSV file
    df.to_csv(data_split_name)

    return df


# change back to working directory
os.chdir('../../')

# define data directory and create mappings of emotion dataset
emotion_dataset = create_csv_audio_mappings(data_directory, 'emotional_data_mappings.csv')

