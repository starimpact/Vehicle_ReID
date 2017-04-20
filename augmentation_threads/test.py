import numpy as np
from ctypes import *
import time
import cv2


func_c = CDLL('./build/libaugment.so')

def test():
  imgnum = 150
  fns = ['/home/mingzhang/Pictures/plates1.jpg']*imgnum
  num = len(fns) 
  strs = (c_char_p*num)()
  strs[:] = fns
  stdH, stdW = 299, 299
  imgsout = np.zeros((imgnum, stdH, stdW, 3), dtype=np.float32)
  print strs
  t0 = time.time()
  func_c.do_augment_threads(strs, num, stdH, stdW, 
                 imgsout.ctypes.data_as(POINTER(c_float))) 
  t1 = time.time()
  print t1-t0
  for i in xrange(imgnum):
    img = imgsout[i]
    cv2.imshow('hi', img)
    cv2.waitKey(0)


if __name__=='__main__':
  test()



