from PIL import Image 
import numpy
from joblib import Parallel, delayed
import os

def criaPatch(nomeImg):
    i = 0
    img = Image.open(r"specsTest/" + nomeImg)
    arr = numpy.array(img)
    left = 0
    top = 0
    right = 8
    bottom = 128
    while(arr.shape[1] - 8 > right):
        im1 = img.crop((left, top, right, bottom)) 
        im1.save('patchsTest/'+ nomeImg.replace('.png', '') + '_' + str(i) + ".png")
        i+=1
        left+=8
        right+=8 

if __name__ == "__main__":
    imagens =  os.listdir('/home/vitor/Documentos/TCC-1/freesound-audio-tagging/specsTest')
    Parallel(n_jobs=4, verbose=1)(delayed(criaPatch)(i) for i in imagens)
