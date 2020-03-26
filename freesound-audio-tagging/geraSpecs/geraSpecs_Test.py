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
    y, sr = librosa.load("../audio_test/" + img, sr=44100)
    D = np.abs(librosa.feature.melspectrogram(y, hop_length=256, n_fft=1024, n_mels=64, fmin=125, fmax=7500))
    fig = plt.figure(figsize=(D.shape[1]/dpi, D.shape[0]/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.pcolormesh(np.log10(D + 0.001))
    plt.savefig("../specsTest/" + img.replace('.wav', '') + '.png')
    plt.close()

readCSV = pd.read_csv("../CSV/test_post_competition.csv")
name = readCSV['fname']
categoria = readCSV['label']
Parallel(n_jobs=6, verbose=10)(delayed(gerarSpecs)(i) for i in name)

    

