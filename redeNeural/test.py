from PIL import Image
import numpy as np
from numpy import array
test = np.loadtxt("nomes.txt", delimiter='\n', dtype= 'str')

for i in test:

    img1 = Image.open(i)
    print(img1)
    arr =  array(img1)
    print(arr.shape)