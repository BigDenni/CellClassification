# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:05:10 2015

@author: dennis
"""
from openslide import OpenSlide
from PIL import Image
import os

#in_slide = '/media/dennis/SAMSUNG/Home/Documents/CellClassification/Data/Ventana/1A.tif'
#in_slide = '/home/dennis/CellClassification/data/1A.tif'
in_slide = '/media/dennis/SAMSUNG/Home/Documents/CellClassification/Data/Aperio/12944.svs'
if os.path.exists(in_slide):
    print 'YES'
    imagefile = OpenSlide(in_slide)
    print imagefile.level_count
    print imagefile.dimensions
    print imagefile.level_dimensions
    print imagefile.level_downsamples
    print imagefile.associated_images
