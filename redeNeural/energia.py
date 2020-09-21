from PIL import Image
from numpy import array
import numpy as np
import dill

def calculaEnergia(img):
    patches = []
    soma = np.sum(np.square(img), axis=0)
    soma[:] = [x / soma[np.argmax(soma)] for x in soma]
    for i in range(0,soma.shape[0]):
        if (soma[i] > 0.5):
            patches.append(int(i/26))
    return set(patches)
    

imagem = Image.open("025f3fd2.png")
arr = array(imagem)
arr = arr[:,:,:3]
arr = np.mean(arr, axis=-1, keepdims=True)
calculaEnergia(arr)