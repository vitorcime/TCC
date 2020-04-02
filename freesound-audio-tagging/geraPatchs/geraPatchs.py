from PIL import Image 
import numpy
from joblib import Parallel, delayed
import os
import sys
def criaPatch(args):
    nomeImg, tipoDado = args
    i = 0
    img = Image.open(r"../specs" +tipoDado + "/" + nomeImg)
    arr = numpy.array(img)
    left = 0
    top = 0
    right = 26
    bottom = 64
    while(arr.shape[1] - 13 > right):
        im1 = img.crop((left, top, right, bottom)) 
        im1.save('../patchs' + tipoDado + '/'+ nomeImg.replace('.png', '') + '_' + str(i) + ".png")
        i+=1
        left+=13
        right+=13 

if(sys.argv[1] == 'train'):
    imagens =  os.listdir('../specsTrain/')
    Parallel(n_jobs=6, verbose=10)(delayed(criaPatch)((i, 'Train')) for i in imagens)
if(sys.argv[1] == 'test'):
    imagens =  os.listdir('../specsTest/')
    Parallel(n_jobs=6, verbose=10)(delayed(criaPatch)((i, 'Test')) for i in imagens)