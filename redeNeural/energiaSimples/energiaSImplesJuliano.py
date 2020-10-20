import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import array
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import copy
import shutil

energia = 0

# TODO: colocar parametro n_frames e tipo_limiar (media, media - std, media - (std/2))

def calculaEnergiaTrain(img, lista): 
    img = img.replace('.wav', '.png')
    imagem = Image.open('../../freesound-audio-tagging/specsTrain/'+img)
    patches = []
    arr = array(imagem)
    arr = arr[:,:,:3]
    arr = np.mean(arr, axis=-1, keepdims=True)
    print(arr.shape)
    soma = np.sum(np.square(arr), axis=0)
    print(soma.shape)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]
    print(soma.shape)

    #passar um filtro passa-baixas para suavizar os "picos" altos e baixos
    k = np.ones(10) / 10
    soma_c = np.convolve(soma[:,0], k)

    #calcular o limiar
    media_soma = np.mean(soma_c)
    std_soma = np.std(soma_c)
    limiar =  media_soma - (std_soma / 2)

    #sel é um vetor de frames "selecionados". Nesse caso, o threshold é np.mean(soma_c). Os frames que excedem o treshold passam a ser 1, enquanto os demais
    #se tornam 0.
    sel = copy.copy(soma_c)
    sel[sel>=limiar] = 1
    sel[sel<1] = 0

    plt.plot(soma)
    plt.plot(soma_c)
    plt.axhline(limiar, color='r')
    plt.plot(sel)
    print(media_soma, std_soma, limiar)
    plt.savefig('energia.png')
    plt.close()
    
    patches = sorted(list(set(filter(None,[ int(i / (26/2)) if sel[i] else None for i in range(soma.shape[0])]))))

    print(patches)

    print(len(patches))

    # for i in range(0,soma.shape[0]):
    #     if (soma[i] >= energia and i+6 <= soma.shape[0]):
    #         patches.append(int(i/26))
    # patches = set(patches)

    # for i in patches:
    #     patch_filename = '../../freesound-audio-tagging/patchsTrain/'+img.replace('.png', '_') + str(i) + '.png'
    #     if not os.path.exists(patch_filename):
    #         continue
    #     shutil.copy(patch_filename, 'selected/' + img.replace('.png', '_') + str(i) + '.png')
    # full_image = '../../freesound-audio-tagging/specsTrain/'+img
    # shutil.copy(full_image, 'selected/' + img)




arquivo = pd.read_csv('../../freesound-audio-tagging/CSV/audiosVerificados.csv')
name = sorted(arquivo['fname']) 
l = []
#calculaEnergiaTrain(name[100], l)
#calculaEnergiaTrain(name[101], l)
calculaEnergiaTrain(name[102], l)