"""
Tristin Johnson
May 2nd, 2022

Script to download and install various splits of LibriSpeech.
"""
import os
import argparse

# get the data split
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir_name', required=True, help='the name of the folder in which to download LibriSpeech. Ex: \'Data\'')
parser.add_argument('--split', required=True, help='which dataset to download: [dev, train-clean-100, train-clean-360, test]')
args = parser.parse_args()

# make a new data directory
os.makedirs('../' + args.data_dir_name)
data_dir = '../' + args.data_dir_name + '/'

# change to data directory
os.chdir(data_dir)

# if 'dev' or validation split
if args.split == 'dev':
    os.system("sudo wget -c http://www.openslr.org/resources/12/dev-clean.tar.gz")
    os.system("tar -xvf dev-clean.tar.gz")

# if 'train-clean' 100 hours
elif args.split == 'train-clean-100':
    os.system("sudo wget -c http://www.openslr.org/resources/12/train-clean-100.tar.gz")
    os.system("tar -xvf train-clean-100.tar.gz")

# if 'train-clean' 360 hours
elif args.split == 'train-clean-360':
    os.system("sudo wget -c http://www.openslr.org/resources/12/train-clean-360.tar.gz")
    os.system("tar -xvf train-clean-360.tar.gz")

# if 'test' split
elif args.split == 'test':
    os.system("sudo wget -c http://www.openslr.org/resources/12/test-clean.tar.gz")
    os.system("tar -xvf test-clean.tar.gz")

else:
    print('Please enter a valid training set')

print('LibriSpeech ' + args.split + ' has successfully been download at: ' + os.getcwd())
