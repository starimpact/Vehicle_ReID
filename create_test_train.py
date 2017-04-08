import os
import numpy as np
import shutil


def create_test_train(srcfn, camfn0, camfn1):
  query_max = 100
  now_idx = 0
  for line in file(srcfn):
    line = line.replace('\n', '')
    parts = line.split(',')
    path = parts[0]
    imgfns = parts[1:]
    rndidxs = np.random.permutation(range(len(imgfns)))
    carid = path.split('/')[-1]
    dstcam0 = camfn0 + '/' + carid
    dstcam1 = camfn1 + '/' + carid
    now_idx += 1 
    if int(carid) < query_max:
      #copy first image into dstcam0
      if not os.path.exists(dstcam0):
        os.makedirs(dstcam0)
      srcfp = path + '/' + imgfns[rndidxs[0]]
      dstfp = dstcam0 + '/' + imgfns[rndidxs[0]]
      shutil.copy(srcfp, dstfp) 
    #copy second image into dstcam1
    if not os.path.exists(dstcam1):
      os.makedirs(dstcam1)
    srcfp = path + '/' + imgfns[rndidxs[1]]
    dstfp = dstcam1 + '/' + imgfns[rndidxs[1]]
    shutil.copy(srcfp, dstfp) 
  return

if __name__=='__main__':
  srcfn = 'data.list'
  camfn0 = 'test_train/cam_0'
  camfn1 = 'test_train/cam_1'
  create_test_train(srcfn, camfn0, camfn1)




