#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   image.py
@Time    :   2023/11/16 14:40:13
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# process default images

# here put the import lib
from PIL import Image
import numpy as np

def cut_image_into_square(
    image_array:np.ndarray,
    cutting_mode:int = 0,
    ) -> np.ndarray:
    # by default, the function cuts the image based on the smaller side (height or width)
    # cutting mode explanation:
    # 0: top or left
    # 1: center
    # 2: bottom or right
    assert cutting_mode in [0,1,2]
    height, width, _ = image_array.shape
    if height>width:
        if cutting_mode == 0:
            return image_array[:width,:,:]
        elif cutting_mode == 1:
            return image_array[(height-width)//2:(height+width)//2,:,:]
        else:
            return image_array[-width:,:,:]
    elif height == width:
        return image_array
    else:
        if cutting_mode == 0:
            return image_array[:,:height,:]
        elif cutting_mode == 1:
            return image_array[:,:height,:]
        else:
            return image_array[:,-height:,:]
    
    
    
if __name__ == "__main__":
    TOPLEFT = 0
    CENTER = 1
    BOTTOMRIGHT = 2
    
    for i in range(1,4):
        # load image through PIL
        image = Image.open('Farm_image_{}/Farm_image_{}_640x360.png'.format(i,i))
        image_array = np.asarray(image)
        output_image = cut_image_into_square(image_array, cutting_mode=CENTER)
        # save image to .png file through PIL
        output_image = Image.fromarray(output_image)
        output_image.save('Farm_image_{}/Farm_image_{}_360x360.png'.format(i,i))