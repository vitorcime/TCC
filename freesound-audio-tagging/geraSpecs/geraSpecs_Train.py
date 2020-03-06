import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display
import pandas as pd 
from joblib import Parallel, delayed
import matplotlib as mpl
mpl.use('Agg')

def gerarSpecs(img):
    dpi = 100.0
    y, sr = librosa.load("audio_train/" + img, sr=44100)
    D = np.abs(librosa.feature.melspectrogram(y, hop_length=256, n_fft=1024, n_mels=128))
    fig = plt.figure(figsize=(D.shape[1]/dpi, D.shape[0]/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(librosa.amplitude_to_db(D, ref=np.max))
    plt.savefig("specsTrain/" + img.replace('.wav', '') + '.png')
    plt.close()

readCSV = pd.read_csv("train_post_competition.csv")
name = readCSV['fname']
categoria = readCSV['label']
Parallel(n_jobs=4, verbose=1)(delayed(gerarSpecs)(i) for i in name)

    

