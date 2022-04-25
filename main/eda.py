# eda.py

# When running this tutorial in Google Colab, install the required packages
# with the following.
#!pip install torchaudio librosa boto3
# C:\\Users\\J\\Documents\\Classes\\Classes_Sp_2022\\Deep_Learning\\Final_project\\Music-Genre-Classification\\HW\\Lib\\site-packages\\_soundfile_data\\libsndfile64bit.dll

import os
from collections import defaultdict

import torch
import torchaudio
import soundfile as sf
import numpy
# import eda_utils as U

'''This method is a sanity check and looks for the basic details of a single random file. '''
def basic_deets():
    print("This is the eda.py file. ============")
    print(torch.__version__)
    print(torchaudio.__version__)
    # print(numpy.__version__)

    # try to load a file
    ex1_waveform: object
    ex1_waveform, sample_rate = torchaudio.load(open(
        r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\genres_original\classical\classical.00001.wav',
        'rb'))
    # fixed issues by adding raw path and 'rb'.
    print("waveform", ex1_waveform.shape, "\n", ex1_waveform)  # note: this 661k value is hard to understand.
    print("sample_rate", sample_rate)  # sample rate x 30 sec = 661500, + 44 = 661544 = diff is 250

    # stats of the soundfile            # source: https://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration
    f = sf.SoundFile(
        r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\genres_original\classical\classical.00001.wav')
    print('samples = {}'.format(f.frames))
    print('sample rate = {}'.format(f.samplerate))
    print('seconds = {}'.format(f.frames / f.samplerate))

    # find the avg vol?
    print("torch.sum(ex1_waveform)", torch.sum(ex1_waveform))


    print("end basic deets =====\n\n")
    #end m

def avg_vol():
    print("This method seeks avg vol by doing torch sum on all genres.")
    path = r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\genres_original'
    genre_avg_vol_dict = defaultdict()
    for genre in os.listdir(path):
        genre_avg_vol_dict[genre] = 0
        genre_sum = 0
        for file in os.listdir(f'{path}/{genre}'):
            print(file)
            ex_waveform, sample_rate = torchaudio.load(open(os.path.join(path, genre, file), 'rb'))
            genre_sum = torch.sum(ex_waveform)
        genre_avg_vol_dict[genre] = genre_sum
    print(genre_avg_vol_dict)



def main():
    # basic_deets()

    avg_vol()



if __name__ == "__main__":
    main()