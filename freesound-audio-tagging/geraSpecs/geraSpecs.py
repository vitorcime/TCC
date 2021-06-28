import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display
import pandas as pd 
from joblib import Parallel, delayed
import matplotlib as mpl
import sys
mpl.use('Agg')

def gerarSpecsTrain(img):
    dpi = 100.0
    y, sr = librosa.load("../audio_train/" + img, sr=44100)
    D = np.abs(librosa.feature.melspectrogram(y, sr=sr, hop_length=256, n_fft=1024, n_mels=64, fmin=125, fmax=7500))
    fig = plt.figure(figsize=(D.shape[1]/dpi, D.shape[0]/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(np.log10(D + np.finfo(D.dtype).eps))
    plt.savefig("../specsTrain/" + img.replace('.wav', '') + '.png')
    plt.close("all")
    


def gerarSpecsTest(img):
    dpi = 100.0
    y, sr = librosa.load("../audio_test/" + img, sr=44100)
    D = np.abs(librosa.feature.melspectrogram(y, sr=sr, hop_length=256, n_fft=1024, n_mels=64, fmin=125, fmax=7500))
    fig = plt.figure(figsize=(D.shape[1]/dpi, D.shape[0]/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(np.log10(D + np.finfo(D.dtype).eps))
    plt.savefig("../specsTest/" + img.replace('.wav', '') + '.png')
    plt.close("all")
    



if(sys.argv[1] == 'train'):
    readCSV = pd.read_csv("../CSV/train_post_competition.csv")
    name = readCSV['fname']
    Parallel(n_jobs=-1, verbose=1)(delayed(gerarSpecsTrain)(i) for i in name)

if(sys.argv[1] == 'test'):
    readCSV = pd.read_csv("../CSV/test_post_competition.csv")
    name = readCSV['fname']
    Parallel(n_jobs=-1, verbose=1)(delayed(gerarSpecsTest)(i) for i in name)
    

