# eda.py

# When running this tutorial in Google Colab, install the required packages
# with the following.
#!pip install torchaudio librosa boto3
# C:\\Users\\J\\Documents\\Classes\\Classes_Sp_2022\\Deep_Learning\\Final_project\\Music-Genre-Classification\\HW\\Lib\\site-packages\\_soundfile_data\\libsndfile64bit.dll

import os
from collections import defaultdict
import operator
from collections import OrderedDict
import pandas as pd
import csv

import torch
from torch import gt
import torchaudio
import soundfile as sf
import numpy
# import eda_utils as U

'''This method is a sanity check and looks for the basic details of a single random file. '''
def basic_deets():
    print("This section shows some basic details of a random wave file. ============")
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
    print("waht does torch.Size return?", type(ex1_waveform.size))
    print("sample_rate", sample_rate)  # sample rate x 30 sec = 661500, + 44 = 661544 = diff is 250

    # stats of the soundfile            # source: https://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration
    f = sf.SoundFile(
        r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\genres_original\classical\classical.00001.wav')
    print('samples = {}'.format(f.frames))
    print('sample rate = {}'.format(f.samplerate))
    print('seconds = {}'.format(f.frames / f.samplerate))

    print("Thus we can find that samples_661,794 = seconds_30.1333 * sample rate_22050. It also shows that we have only\n"
          "the section of the file that encodes the music data and that the metadata and formatting section has been removed.")

    # find the avg vol?
    # print("torch.sum(ex1_waveform)", torch.sum(ex1_waveform))


    print("end basic deets =====\n\n")
    #end m

def avg_vol():
    print("This section seeks avg vol by doing torch sum on all genres. =============")
    print("We create a tensor of the actual data section of the wav file, use relu to remove negatives and this leaves us positives\n"
          "between 0-1.0. We sum those up, then divide by half of the samples in that file, to get near an average of the positive\n"
          "amplitude. Assumptively, the different use of 'compression' which compresses the volumes differentials in a file are used\n"
          "when creating files of different genres. Eg. classical would tend to use none, resulting in more low volumes and a lower \n"
          "positive amplitude.")
    path = r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\genres_original'
    genre_avg_vol_dict = {}
    for genre in os.listdir(path):
        genre_sum = 0
        for file in os.listdir(f'{path}/{genre}'):
            ##print(file)
            ex_waveform, sample_rate = torchaudio.load(open(os.path.join(path, genre, file), 'rb'))
            x = ex_waveform
            torch.nn.functional.relu(x, inplace=True)
            # print(x)
            # # y = torch.ge(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5]]))
            # x[x:gt(x, torch.tensor([[5]]) )] = 0
            # x = torch.Tensor.apply_ge
            # print(x)
            # print(len(x))
            genre_sum = torch.IntTensor.item( torch.sum(x) * 2 / x.shape[1] )
        genre_avg_vol_dict[genre] = genre_sum

    # genre_avg_vol_dict = sorted(genre_avg_vol_dict)

    sorted_d = dict(sorted(genre_avg_vol_dict.items(), key=operator.itemgetter(1), reverse=False))
    # would a softmax help?
    print("Sorted genre list in order of volume, asc:\n", sorted_d)
    print("The ordering here at the front of the list is expected based on the amount of 'compression' that is generally used in creating wav files for these genres. ")


def get_all_files_avg_vol():
    print("This section gets the avg vols for each song, saves it into a csv.")
    path = r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\genres_original'
    avgVol = [] # list of each song's avg vol
    genreList = []
    for genre in os.listdir(path):
        # genreList.append(genre)
        genre_count = 0
        for file in os.listdir(f'{path}/{genre}'):
            # print(file)
            ex_waveform, sample_rate = torchaudio.load(open(os.path.join(path, genre, file), 'rb'))
            x = ex_waveform
            torch.nn.functional.relu(x, inplace=True)
            piece_avg_vol = torch.IntTensor.item(torch.sum(x) * 2 / x.shape[1]) #w_size = list(w.shape);#print(w_size); # use w_size[0]           # this gets the shape of 1d tensor
            avgVol.append(piece_avg_vol)
            genre_count+=1
            if file == "jazz.00053.wav": # for the missing file, just use the prev's.
                # print("added extra")
                avgVol.append(piece_avg_vol)
                genre_count += 1
        # print(genre, genre_count)
    print("avgVol", len(avgVol), "\n", avgVol)
    # print(genreList)
    return avgVol

    # determine why we only have 999.
    # we removed jazz54
    # load csv into dataframe.
    # add avgvol
    # export back out to new csv.

    def inspect_rms(filepath: str):
        """
        :param str filepath:
        :rtype: tuple[torch.Tensor, torch.Tensor, int]
        """
        features_df = pd.read_csv(filepath)
        length = len(features_df.index)

        # First drop the file name, it is not needed.
        labels = features_df['labels']
        rms_mean = features_df['rms_mean']
        rms_var = features_df['rms_var']

        # Save the names of the categories
        cats = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

        # First save the label column to a separate variable, which can then be turned into onehot Tensor.
        # labels_df = features_df['label']
        # labels = makeOnehotTensorFromDataframe(cats, labels_df)

        # Then drop the label column and turn the features into a Tensor.
        # del features_df['label']
        # # normalization
        # for col in features_df.columns:
        #     features_df[col] = (features_df[col] - min(features_df[col])) / (
        #                 max(features_df[col]) - min(features_df[col]))
        # features = torch.as_tensor(features_df.values, dtype=torch.float32)
        # return features, labels, length

def addCSV(song_avg_vol, filepath: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    :param str filepath:
    :rtype: tuple[torch.Tensor, torch.Tensor, int]
    """
    # features_df = pd.read_csv(filepath)
    # length = len(features_df.index)
    # print(length)
    # print(features_df)

    with open(r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\features_30_sec.csv', 'r') as csvinput:
        with open(r"C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\new_features_30_sec.csv", 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('avg_vol')
            all.append(row)


            for i, row in enumerate(reader):
                row.append(song_avg_vol[i])
                all.append(row)

            writer.writerows(all)

def main():
    print("This is the eda.py file. ============")
    # basic_deets()

    avg_vol()
    song_avg_vol = get_all_files_avg_vol()
    # addCSV(song_avg_vol, r"C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\new_features_30_sec.csv")
    inspect_rms(r'C:\Users\J\Documents\Classes\Classes_Sp_2022\Deep_Learning\Final_project\Music-Genre-Classification\data\features_30_sec.csv')


if __name__ == "__main__":
    main()