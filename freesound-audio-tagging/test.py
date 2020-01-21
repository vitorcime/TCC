import numpy as np
import os
from PIL import Image 
names = os.listdir('patchsTrain/')
imagens_treino = []

for i in names[0:300]:
    arr = Image.open(r"patchsTrain/" + i)
    arr = np.array(arr)
    imagens_treino.append(arr)
imagens_treino = np.asarray(imagens_treino)
print(imagens_treino.shape)