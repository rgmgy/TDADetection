# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

img_dir = '/Users/gy/Desktop/paperPicture/combine/picture/modify0/filter/jpg'

def combine(dir):
    i = 0
    for path in os.listdir(dir):
        if path != ".DS_Store":
            i += 1
            img_path = os.path.join('%s/%s' % (img_dir, path))
            print img_path
            img = Image.open(img_path)
            if i < 19:

                plt.subplot(2, 9, i)
                plt.imshow(img)
                plt.axis('off')
            '''
            else:
                plt.subplot(2, 9, i-9)
                plt.imshow(img)
                plt.axis('off')
            '''


    plt.show()

combine(img_dir)