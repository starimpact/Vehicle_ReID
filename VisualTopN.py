import numpy as np
import cPickle
import cv2
import draw
import os

def visualTopN(fn, qpath, mpath, savepath, showtopn=20):
  stdsize=(200, 200)
  topn = cPickle.load(open(fn, 'rb'))
  for onecar in topn:
    car_id = onecar['id']
    car_name = onecar['name'].split('.bin')[0]
    car_topn = onecar['data']
    qfn = qpath + '/' + car_id + '/' + car_name
    qimg = cv2.imread(qfn) 
    qimg = cv2.resize(qimg, (stdsize[1], stdsize[0]))
    qimg = draw.drawText_Color(qimg, car_id, (0, 0), 20, (0, 0, 255))
    allimg = qimg
#    print car_id, car_name
    idxnow = 0
    for mcar in car_topn:
      idxnow+=1
      m_id = mcar['id']
      m_name = mcar['name'].split('.bin')[0]
      m_score = mcar['score']
      mfn = mpath + '/' + m_id + '/' + m_name
      mimg = cv2.imread(mfn) 
      mimg = cv2.resize(mimg, (stdsize[1], stdsize[0]))
      txt = m_id + '(%.2f)'%(m_score)
      mimg = draw.drawText_Color(mimg, txt, (0, 0), 20, (0, 0, 255))
      allimg = np.append(allimg, mimg, axis=1) 
      if idxnow>=showtopn:
        break
#      print m_id, m_name, m_score
    if car_id==car_topn[0]['id']:
      savefn = savepath+'/right/'
      if not os.path.exists(savefn):
        os.makedirs(savefn)
      savefn += car_id+'='+car_name+'.result.jpg'
    else:
      savefn = savepath+'/wrong/'
      if not os.path.exists(savefn):
        os.makedirs(savefn)
      savefn += car_id+'='+car_name+'.result.jpg'
    print savefn
    cv2.imwrite(savefn, allimg)
  print 'visualization is over ...'

def group2one(imglist, colnum):
  qimg = imglist[0][0]
  gndimgs = imglist[0][1:]
  imglist = imglist[1:]
  imgh, imgw, ch = qimg.shape
  imgnum = len(imglist)
  rownum = int(np.ceil(imgnum*1.0/colnum))
  allimg = np.zeros((imgh*rownum, imgw*(colnum+1), 3), dtype=qimg.dtype)
  allimg[:imgh, :imgw] = qimg 
  hpos = 1
  for gndimg in gndimgs:
    if hpos >= rownum:
      break
    allimg[imgh*hpos:(hpos+1)*imgh, :imgw] = gndimg
    hpos += 1
  for ri in xrange(rownum):
    for ci in xrange(1, colnum+1):
      nowidx = ri * rownum + ci - 1
      if nowidx >= imgnum:
        continue
      allimg[ri*imgh:(ri+1)*imgh, ci*imgw:(ci+1)*imgw] = imglist[nowidx]
  allimg[:, imgw-2:imgw+2, 0] = 0
  allimg[:, imgw-2:imgw+2, 1] = 0
  allimg[:, imgw-2:imgw+2, 2] = 255
#  cv2.imshow('hi', allimg)
#  cv2.waitKey(0)
  return allimg
   

def visualTopN2(fn, savepath, showtopn=20):
  stdsize=(200, 200)
  topn = cPickle.load(open(fn, 'rb'))
  for onecar in topn:
    car_id = str(onecar['id'])
    car_name = onecar['path'].split('/')[-1]
    car_topn = onecar['data']
    qfn = onecar['path']
    qimg = cv2.imread(qfn) 
    qimg = cv2.resize(qimg, (stdsize[1], stdsize[0]))
    qimg = draw.drawText_Color(qimg, car_id, (0, 0), 20, (0, 0, 255))
    qtmps = [qimg]
    for gndfn in onecar['gpath']:
      gndimg = cv2.imread(gndfn) 
      gndimg = cv2.resize(gndimg, (stdsize[1], stdsize[0]))   
      gndimg = draw.drawText_Color(gndimg, 'GND_'+car_id, (0, 0), 20, (0, 0, 255))
      qtmps += [gndimg]
    allimg = [qtmps]
#    print car_id, car_name
    idxnow = 0
    for mcar in car_topn:
      idxnow+=1
      m_id = str(mcar['id'])
      m_name = mcar['path'].split('/')[-1]
      m_score = mcar['score']
      mfn = mcar['path']
      mimg = cv2.imread(mfn) 
      mimg = cv2.resize(mimg, (stdsize[1], stdsize[0]))
      txt = m_id + '(%.2f)'%(m_score)
      mimg = draw.drawText_Color(mimg, txt, (0, 0), 20, (0, 0, 255))
      allimg.append(mimg)
      if idxnow>=showtopn:
        break
#      print m_id, m_name, m_score
    if car_id==str(car_topn[0]['id']):
      savefn = savepath+'/right/'
      if not os.path.exists(savefn):
        os.makedirs(savefn)
      savefn += car_id+'='+car_name+'.result.jpg'
    else:
      savefn = savepath+'/wrong/'
      if not os.path.exists(savefn):
        os.makedirs(savefn)
      savefn += car_id+'='+car_name+'.result.jpg'
    print savefn
    allimg = group2one(allimg, 10)
    cv2.imwrite(savefn, allimg)
  print 'visualization is over ...'
 
if __name__=='__main__':
  topnfn = 'top20.test.bin'
  savepath = 'ImageResult'
  father = '/home/mingzhang/data/car_ReID_for_zhangming/test'
  query_path = father + '/cam_0'
  match_path = father + '/cam_1'
  showtopn = 10
#  visualTopN(topnfn, query_path, match_path, savepath, showtopn)

  topnfn = 'topN_front.bin'
  savepath = 'ImageResult/front'
  showtopn = 50
  visualTopN2(topnfn, savepath, showtopn)

  topnfn = 'topN_back.bin'
  savepath = 'ImageResult/back'
  showtopn = 50
  visualTopN2(topnfn, savepath, showtopn)


