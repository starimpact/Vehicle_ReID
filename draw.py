# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:50:30 2016

@author: mingzhang
"""

import Image, ImageDraw
import ImageFont
import numpy as np

fontfile = 'Font-Simplified-Chinese.ttf'
def drawText(cimg, txt, posxy, fz):
    ttFont0 = ImageFont.truetype(fontfile, fz)
    im = Image.fromarray(cimg, 'RGB')
    drawable = ImageDraw.Draw(im)
    drawable.text ((posxy[0], posxy[1]), txt, fill=(0, 255, 0), font=ttFont0)
    npimg = np.asarray(im)
    
    return npimg


def drawText_Color(cimg, txt, posxy, fz, color):
    ttFont0 = ImageFont.truetype(fontfile, fz)
    im = Image.fromarray(cimg, 'RGB')
    drawable = ImageDraw.Draw(im)
    drawable.text((posxy[0], posxy[1]), txt, fill=color, font=ttFont0)
    npimg = np.asarray(im)
    
    return npimg


def drawText_BKG(cimg, txt, posxy, fz, bkglen):
    ttFont0 = ImageFont.truetype(fontfile, fz)
    im = Image.fromarray(cimg, 'RGB')
    drawable = ImageDraw.Draw(im)
    drawable.polygon(((posxy[0], posxy[1]), \
                      (posxy[0]+bkglen, posxy[1]), \
                      (posxy[0]+bkglen, posxy[1]+fz), \
                      (posxy[0], posxy[1]+fz)), fill=(255, 255, 255))
    drawable.text ((posxy[0], posxy[1]), txt, fill=(0, 0, 255), font=ttFont0)
    npimg = np.asarray(im)
    
    return npimg
    