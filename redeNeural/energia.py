from PIL import Image
from numpy import array
import numpy as np
from sklearn.preprocessing import StandardScaler
import dill

def calculaEnergia(img):
    soma = np.sum(np.square(img), axis=0)
    ss = StandardScaler()
    ss.fit(soma)
    dill.dump(ss, open('ss.dill','wb'))
    soma = ss.transform(soma)
    print(soma.shape)
            

imagem = Image.open("025f3fd2.png")
arr = array(imagem)
print(arr.shape)
arr = arr[:,:,:3]
arr = np.mean(arr, axis=-1, keepdims=True)
print(arr.shape)
calculaEnergia(arr)