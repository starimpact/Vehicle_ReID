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
    allimg = qimg
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
      allimg = np.append(allimg, mimg, axis=1) 
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
  showtopn = 10
  visualTopN2(topnfn, savepath, showtopn)

  topnfn = 'topN_back.bin'
  savepath = 'ImageResult/back'
  showtopn = 10
  visualTopN2(topnfn, savepath, showtopn)


