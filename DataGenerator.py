import numpy as np
import cv2
import cPickle
import os
import mxnet as mx
import time

datafn = '/media/data1/mzhang/data/car_ReID_for_zhangming/data/data.list'

def get_datalist(datafn):
  datafile = open(datafn, 'r')
  datalist = datafile.readlines()
  datalen = len(datalist)
  for di in xrange(datalen):
    datalist[di] = datalist[di].replace('\n', '')
    
  datafile.close()
  
  return datalist


def get_datalist2(datafn_list):
  datalist = []
  for datafn in datafn_list:
    datalist += get_datalist(datafn) 
  return datalist


def get_proxyset(proxyfn, proxyshape):
  proxy_set = []
  if os.path.isfile(proxyfn):
    print 'loading proxy set from ', proxyfn
    proxy_set = cPickle.load(open(proxyfn, 'rb'))
    assert(proxy_set.shape==proxyshape)
    return proxy_set

  print 'creating proxy set to ', proxyfn
  p = np.random.rand(proxyshape[0], proxyshape[1]) - 0.5 
  p = p.astype(dtype=np.float32)
  if True:
    pn = np.sqrt(np.sum(p*p, axis=1))
    pn = np.reshape(pn, (pn.shape[0], 1))
    proxy_set = p / pn * 4.0
  else:
    proxy_set = p
  cPickle.dump(proxy_set, open(proxyfn, 'wb'));
  return proxy_set


def get_pairs_data_label(data_infos, label_infos, datalist, data_rndidx, batch_now):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  neednum = batchsize / 2 + 1
  if (batch_now+1)*neednum > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*neednum:(batch_now+1)*neednum]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['sons'] = parts[1:]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  same_num = batchsize / 2
  diff_num = batchsize - same_num
  dataidx = 0
  datas = {}
  labels = {}
  datas['part1_data'] = np.zeros(data_infos[0][1], dtype=np.float32)
  datas['part2_data'] = np.zeros(data_infos[1][1], dtype=np.float32)
  labels['label'] = np.zeros(label_infos[0][1], dtype=np.float32)
  #ready same data
  for si in xrange(same_num):
    onecar = cars[si]
    carpath = onecar['path']
    carsons = onecar['sons']
    rndidx = np.random.permutation(len(carsons))
    tmpath = carpath+'/'+carsons[rndidx[0]]
    son0 = cv2.imread(tmpath)
#    print 0, tmpath, son0.shape, stdsize
    stdson0 = cv2.resize(son0, (stdsize[1], stdsize[0]))
    stdson0 = stdson0.astype(np.float32) / 255.0
    tmpath = carpath+'/'+carsons[rndidx[1]]
    son1 = cv2.imread(tmpath)
#    print 1, tmpath, son1.shape, stdsize
    stdson1 = cv2.resize(son1, (stdsize[1], stdsize[0]))
    stdson1 = stdson1.astype(np.float32) / 255.0
    datas['part1_data'][dataidx, 0] = stdson0[:, :, 0]
    datas['part1_data'][dataidx, 1] = stdson0[:, :, 1]
    datas['part1_data'][dataidx, 2] = stdson0[:, :, 2]
    datas['part2_data'][dataidx, 0] = stdson1[:, :, 0]
    datas['part2_data'][dataidx, 1] = stdson1[:, :, 1]
    datas['part2_data'][dataidx, 2] = stdson1[:, :, 2]
    labels['label'][dataidx] = 1
    dataidx += 1

  #ready diff data
  for si in xrange(diff_num):
    rndidx = np.random.permutation(len(cars))
    car0 = cars[rndidx[0]]
    car0len = len(car0['sons'])
    car1 = cars[rndidx[1]]
    car1len = len(car1['sons'])
    son0 = cv2.imread(car0['path']+'/'+car0['sons'][np.random.randint(0, car0len)])
    stdson0 = cv2.resize(son0, (stdsize[1], stdsize[0]))
    stdson0 = stdson0.astype(np.float32) / 255.0
    son1 = cv2.imread(car1['path']+'/'+car1['sons'][np.random.randint(0, car1len)])
    stdson1 = cv2.resize(son1, (stdsize[1], stdsize[0]))
    stdson1 = stdson1.astype(np.float32) / 255.0
    datas['part1_data'][dataidx, 0] = stdson0[:, :, 0]
    datas['part1_data'][dataidx, 1] = stdson0[:, :, 1]
    datas['part1_data'][dataidx, 2] = stdson0[:, :, 2]
    datas['part2_data'][dataidx, 0] = stdson1[:, :, 0]
    datas['part2_data'][dataidx, 1] = stdson1[:, :, 1]
    datas['part2_data'][dataidx, 2] = stdson1[:, :, 2]
    labels['label'][dataidx] = -1
    dataidx += 1

  return datas, labels


def get_data_label_test(data_shape, datalist, which_car
                        , normalize=True):
  """
  data_shape: (chn, h, w)
  datalist: string list
  which_car: query which car
  """
  query_line = datalist[which_car]
  onecar = {}
  parts = query_line.split(',')
  onecar['path'] = parts[0]
  onecar['sons'] = parts[1:]
  num_sons = len(onecar['sons'])
  parts2 = onecar['path'].split('/')
  onecar['id'] = parts2[-1]
  stdsize = data_shape[1:]
#  print data_shape, stdsize
  carinfo = {}
  carinfo['id'] = onecar['id']
  carinfo['sons'] = []
  for si in xrange(num_sons):
    queryone = onecar['sons'][si]
    tmppath = onecar['path'] + '/' + queryone
    sonimg = cv2.imread(tmppath)  
    stdson = cv2.resize(sonimg, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
    if normalize:
      stdson = get_normalization(stdson)
    stdson_tmp = np.zeros((3,)+stdsize, dtype=np.float32)
    stdson_tmp[0] = stdson[:, :, 0]
    stdson_tmp[1] = stdson[:, :, 1]
    stdson_tmp[2] = stdson[:, :, 2]
    soninfo = {}
    soninfo['name'] = queryone
    soninfo['data'] = stdson_tmp
    carinfo['sons'].append(soninfo)
  
  return carinfo


def get_feature_label_test__(data_shape, datalist, which_car):
  """
  data_shape: (chn, h, w)
  datalist: string list
  which_car: query which car
  """
  query_line = datalist[which_car]
  onecar = {}
  parts = query_line.split(',')
  onecar['path'] = parts[0]
  onecar['sons'] = parts[1:]
  num_sons = len(onecar['sons'])
  parts2 = onecar['path'].split('/')
  onecar['id'] = parts2[-1]
  stdsize = data_shape[1:]
#  print data_shape, stdsize
  carinfo = {}
  carinfo['id'] = onecar['id']
  carinfo['sons'] = []
  for si in xrange(num_sons):
    queryone = onecar['sons'][si]
    tmppath = onecar['path'] + '/' + queryone
    sonfeat = cPickle.load(open(tmppath, 'rb'))
    soninfo = {}
    soninfo['name'] = queryone
    soninfo['data'] = sonfeat.reshape(data_shape)
    carinfo['sons'].append(soninfo)
  
  return carinfo


def get_feature_label_query_test(data_shape, datalist, which_idx):
  """
  data_shape: (batch, chn, h, w)
  datalist: string list
  which_idx: query a batch 
  """
  batchsize = data_shape[0]
  query_line = datalist[which_idx]
  batch_info = {'paths':[], 'ids':[], 'names':[], 'data':np.zeros(data_shape, dtype=np.float32)}
  for qli in xrange(batchsize):
    parts = query_line.split(',')
    pathpre = parts[0]
    namenow = parts[1]
    pathnow = pathpre + '/' + namenow
    parts2 = pathpre.split('/')
    idnow = parts2[-1]
    featnow = cPickle.load(open(pathnow, 'rb'))
    batch_info['data'][qli] = featnow[0]
    batch_info['paths'].append(pathnow)
    batch_info['ids'].append(idnow)
    batch_info['names'].append(namenow)
  
  return batch_info


def get_feature_label_test(data_shape, datalist, which_batch):
  """
  data_shape: (batch, chn, h, w)
  datalist: string list
  which_batch: query a batch 
  """
  batchsize = data_shape[0]
  query_batch = datalist[which_batch*batchsize:(which_batch+1)*batchsize]
  batch_info = {'paths':[], 'ids':[], 'names':[], 'data':np.zeros(data_shape, dtype=np.float32)}
  for qli, query_line in enumerate(query_batch):
    parts = query_line.split(',')
    pathpre = parts[0]
    namenow = parts[1]
    pathnow = pathpre + '/' + namenow
    parts2 = pathpre.split('/')
    idnow = parts2[-1]
    featnow = cPickle.load(open(pathnow, 'rb'))
    batch_info['data'][qli] = featnow[0]
    batch_info['paths'].append(pathnow)
    batch_info['ids'].append(idnow)
    batch_info['names'].append(namenow)
  
  return batch_info


def get_pairs_data_label_rnd(data_infos, label_infos):
  datas = {}
  for dinfo in data_infos:
#    datas[dinfo[0]] = np.zeros(dinfo[1])
    datas[dinfo[0]] = np.random.rand(*dinfo[1])
  labels = {}
  for linfo in label_infos:
#    labels[linfo[0]] = np.zeros(linfo[1])
    rndlabels = np.random.binomial(1, 0.5, linfo[1])
    rndlabels[rndlabels==0] = -1
    labels[linfo[0]] = rndlabels
#    print rndlabels

  return datas, labels


def get_data_label(data_infos, label_infos, datalist, data_rndidx, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*batchsize:(batch_now+1)*batchsize]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[0].split('/')[-1]
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  datas = {}
  labels = {}
  datas['data'] = np.zeros(data_infos[0][1], dtype=np.float32)
  labels['label'] = np.zeros(label_infos[0][1], dtype=np.float32)
  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carson = onecar['son']
    tmpath = carpath+'/'+carson
    son = cv2.imread(tmpath)
    if rndcrop:
      son = get_rnd_crop(son)
#    print 0, tmpath, son0.shape, stdsize
    stdson = cv2.resize(son, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
    if rndcont:
      stdson = get_rnd_contrast(stdson)
    if rndnoise:
      stdson = get_rnd_noise(stdson)
    if normalize:
      stdson = get_normalization(stdson)
    if rndrotate:
      stdson = get_rnd_rotate(stdson)
    if rndhflip:
      stdson = get_rnd_hflip(stdson)
#    print carid, stdson
    datas['data'][si, 0] = stdson[:, :, 0]
    datas['data'][si, 1] = stdson[:, :, 1]
    datas['data'][si, 2] = stdson[:, :, 2]
    labels['label'][si] = carid
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)

  return datas, labels


def get_data_label_proxy(data_infos, label_infos, datalist, data_rndidx, proxyset, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*batchsize:(batch_now+1)*batchsize]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[0].split('/')[-1]
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  datas = {}
  labels = {}
  datas['data'] = np.zeros(data_infos[0][1], dtype=np.float32)
  labels['proxy_yM'] = np.zeros(label_infos[0][1], dtype=np.float32)
#  labels['proxy_Z'] = np.zeros(label_infos[1][1], dtype=np.float32)
  labels['proxy_ZM'] = np.ones(label_infos[1][1], dtype=np.float32)
  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carson = onecar['son']
    tmpath = carpath+'/'+carson
    son = cv2.imread(tmpath)
    if rndcrop:
      son = get_rnd_crop(son)
#    print 0, tmpath, son0.shape, stdsize
    stdson = cv2.resize(son, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
    if rndcont:
      stdson = get_rnd_contrast(stdson)
    if rndnoise:
      stdson = get_rnd_noise(stdson)
    if normalize:
      stdson = get_normalization(stdson)
    if rndrotate:
      stdson = get_rnd_rotate(stdson)
    if rndhflip:
      stdson = get_rnd_hflip(stdson)
#    print carid, stdson
    datas['data'][si, 0] = stdson[:, :, 0]
    datas['data'][si, 1] = stdson[:, :, 1]
    datas['data'][si, 2] = stdson[:, :, 2]
    labels['proxy_yM'][si, carid] = 1
#    labels['proxy_Z'][:] = proxyset
    labels['proxy_ZM'][si, carid] = 0
#    print proxyset[carid], np.sum(proxyset[carid] * proxyset[carid])
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)

  datas = [mx.nd.array(datas['data'])]
  labels = [mx.nd.array(labels['proxy_yM']), mx.nd.array(labels['proxy_ZM'])]

  return datas, labels




def get_rnd_crop(img):
  imgh, imgw = img.shape[:2]
  rndmg = np.random.randint(0, 10, 4)/100.0
  mgs = [imgh*rndmg[0], imgh*rndmg[1], imgw*rndmg[2], imgw*rndmg[3]]
  cropimg = img[mgs[0]:imgh-mgs[1], mgs[2]:imgw-mgs[3]]
  return cropimg


def get_rnd_contrast(img):
  rndv = np.random.randint(75, 100)/100.0
  img *= rndv

  return img

def get_rnd_noise(img):
  rndv = np.random.randint(1, 20) / 1000.0
  gauss = np.random.normal(0, rndv, img.shape)
  img += gauss
  img[img<0] = 0
  img[img>1.0] = 1.0

  return img


def get_rnd_rotate(img):
  imgh, imgw = img.shape[:2]
  rndv = np.random.randint(-30, 30)#*3.1415/180
#  print rndv
  rotmat = cv2.getRotationMatrix2D((imgw/2, imgh/2), rndv, 1.0)
  rotimg = cv2.warpAffine(img, rotmat, (imgw, imgh))

  return rotimg


def get_rnd_hflip(img):
  rndv = np.random.rand()
  hfimg = img
  if rndv<0.5:
    hfimg = img[:, ::-1]

  return hfimg


def get_normalization_rgb(img):
  mean = img.mean(axis=(0, 1))
  std = img.std(axis=(0, 1))
  nimg = (img-mean)/std

  return nimg


def get_normalization(img):
  mean = img.mean()
  std = img.std()
  nimg = (img-mean)/std

  return nimg



  


#format: path,imgname
def get_data_label_proxy_mxnet(data_infos, label_infos, datalist, data_rndidx, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*batchsize:(batch_now+1)*batchsize]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[0].split('/')[-1]
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  datas = {}
  labels = {}
  datas['data'] = np.zeros(data_infos[0][1], dtype=np.float32)
  labels['proxy_yM'] = np.zeros(label_infos[0][1], dtype=np.float32)
  labels['proxy_ZM'] = np.ones(label_infos[1][1], dtype=np.float32)
  
  imgs = []
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carson = onecar['son']
    tmpath = carpath+'/'+carson
#    son = mx.image.imdecode(open(tmpath).read())
    son = cv2.imread(tmpath)
    imgs.append(son)

  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carid = int(onecar['id'])
   
    son = imgs[si] 
#    son = son.asnumpy()
#    print son.shape
    if rndcrop:
      son = get_rnd_crop(son)
#    print 0, tmpath, son0.shape, stdsize
    stdson = cv2.resize(son, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
    if rndcont:
      stdson = get_rnd_contrast(stdson)
    if rndnoise:
      stdson = get_rnd_noise(stdson)
    if normalize:
      stdson = get_normalization(stdson)
    if rndrotate:
      stdson = get_rnd_rotate(stdson)
    if rndhflip:
      stdson = get_rnd_hflip(stdson)
#    print carid, stdson
    datas['data'][si, 0] = stdson[:, :, 0]
    datas['data'][si, 1] = stdson[:, :, 1]
    datas['data'][si, 2] = stdson[:, :, 2]
    labels['proxy_yM'][si, carid] = 1
    labels['proxy_ZM'][si, carid] = 0
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)
  datas_nd = [mx.nd.array(datas['data'])]
  label_nd = [mx.nd.array(labels['proxy_yM']), mx.nd.array(labels['proxy_ZM'])]
  return datas_nd, label_nd


from ctypes import *
func_c = CDLL('./augmentation_threads/libaugment.so')
def aug_threads_c(paths, tmpshape):
  imgnum, chs, stdH, stdW = tmpshape 
  imgsout = np.zeros((imgnum, stdH, stdW, chs), dtype=np.float32)
  strs = (c_char_p*imgnum)()
  strs[:] = paths
#  t0 = time.time()
  func_c.do_augment_threads(strs, imgnum, stdH, stdW,
                 imgsout.ctypes.data_as(POINTER(c_float)))
#  t1 = time.time()
#  print t1-t0
#  for i in xrange(imgnum):
#    img = imgsout[i]
#    cv2.imshow('hi', img)
#    cv2.waitKey(0)
  return imgsout


def aug_threads_c2(paths, tmpshape, imgsout):
  imgnum, chs, stdH, stdW = tmpshape 
  strs = (c_char_p*imgnum)()
  strs[:] = paths
#  t0 = time.time()
  func_c.do_augment_threads(strs, imgnum, stdH, stdW,
                 imgsout.ctypes.data_as(POINTER(c_float)))
#  t1 = time.time()
#  print t1-t0
#  for i in xrange(imgnum):
#    img = imgsout[i]
#    img = img.swapaxes(0, 1)
#    img = img.swapaxes(1, 2)
#    cv2.imshow('hi', img)
#    cv2.waitKey(0)


#format: path,imgname
def get_data_label_proxy_mxnet_threads(data_infos, datas, label_infos, labels, datalist, data_rndidx, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*batchsize:(batch_now+1)*batchsize]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[0].split('/')[-1]
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  labels['proxy_yM'][:] = 0
  labels['proxy_ZM'][:] = 1.0
  
  tmpaths = []
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carson = onecar['son']
    tmpath = carpath+'/'+carson
    tmpaths.append(tmpath)
  
  aug_data = datas['databuffer']
  aug_threads_c2(tmpaths, data_infos[0][1], aug_data)
  datas['data'][:] = aug_data

  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carid = int(onecar['id'])
    labels['proxy_yM'][si, carid] = 1
    labels['proxy_ZM'][si, carid] = 0
  datas_nd = [datas['data']]
  label_nd = [labels['proxy_yM'], labels['proxy_ZM']]
  return datas_nd, label_nd




#format: path,imgname,idnumber
def get_data_label_proxy_mxnet2(data_infos, label_infos, datalist, data_rndidx, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*batchsize:(batch_now+1)*batchsize]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[2]
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  datas = {}
  labels = {}
  datas['data'] = np.zeros(data_infos[0][1], dtype=np.float32)
  labels['proxy_yM'] = np.zeros(label_infos[0][1], dtype=np.float32)
  labels['proxy_ZM'] = np.ones(label_infos[1][1], dtype=np.float32)
  
  imgs = []
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carson = onecar['son']
    tmpath = carpath+'/'+carson
#    son = mx.image.imdecode(open(tmpath).read())
    son = cv2.imread(tmpath)
    imgs.append(son)

  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carid = int(onecar['id'])
   
    son = imgs[si] 
#    son = son.asnumpy()
#    print son.shape
    if rndcrop:
      son = get_rnd_crop(son)
#    print 0, tmpath, son0.shape, stdsize
    stdson = cv2.resize(son, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
    if rndcont:
      stdson = get_rnd_contrast(stdson)
    if rndnoise:
      stdson = get_rnd_noise(stdson)
    if normalize:
      stdson = get_normalization(stdson)
    if rndrotate:
      stdson = get_rnd_rotate(stdson)
    if rndhflip:
      stdson = get_rnd_hflip(stdson)
#    print carid, stdson
    datas['data'][si, 0] = stdson[:, :, 0]
    datas['data'][si, 1] = stdson[:, :, 1]
    datas['data'][si, 2] = stdson[:, :, 2]
    labels['proxy_yM'][si, carid] = 1
    labels['proxy_ZM'][si, carid] = 0
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)
  datas_nd = [mx.nd.array(datas['data'])]
  label_nd = [mx.nd.array(labels['proxy_yM']), mx.nd.array(labels['proxy_ZM'])]
  return datas_nd, label_nd


#format: path,imgname,idnumber
def get_data_label_proxy_mxnet2_threads(data_infos, label_infos, datalist, data_rndidx, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  for idx in data_rndidx[batch_now*batchsize:(batch_now+1)*batchsize]:
    data_batch.append(datalist[idx])
  cars = []
  for onedata in data_batch:
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[2]
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  datas = {}
  labels = {}
  datas['data'] = mx.nd.zeros(data_infos[0][1], dtype=np.float32)
  labels['proxy_yM'] = mx.nd.zeros(label_infos[0][1], dtype=np.float32)
  labels['proxy_ZM'] = mx.nd.ones(label_infos[1][1], dtype=np.float32)
  
  tmpaths = []
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carson = onecar['son']
    tmpath = carpath+'/'+carson
    tmpaths.append(tmpath)
 
#  t0 = time.time() 
  aug_data = aug_threads_c(tmpaths, data_infos[0][1])
#  t1 = time.time() 

  aug_data = aug_data.swapaxes(2, 3)
  datas['data'][:] = aug_data.swapaxes(1, 2)

  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carid = int(onecar['id'])
   
    labels['proxy_yM'][si, carid] = 1
    labels['proxy_ZM'][si, carid] = 0
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)
#  t2 = time.time() 
  datas_nd = [datas['data']]
  label_nd = [labels['proxy_yM'], labels['proxy_ZM']]
#  print t2-t1, t1-t0
  return datas_nd, label_nd



#format: path,imgname
def get_data_label_proxy_batch_mxnet(data_infos, label_infos, datalist, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
  data_batch = []
  idlist = []
  batch_info = []
  for idx in xrange(batch_now*batchsize, (batch_now+1)*batchsize):
    data_batch.append(datalist[idx])
    idlist.append(idx)
  cars = []
  for idx, onedata in zip(idlist, data_batch):
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[-1] 
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)
    oneinfo = '%s,%s,%s'%(parts[0], parts[1], parts[2])
    batch_info.append(oneinfo)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  datas = {}
  labels = {}
  datas['data'] = np.zeros(data_infos[0][1], dtype=np.float32)
  labels['proxy_yM'] = np.zeros(label_infos[0][1], dtype=np.float32)
  labels['proxy_ZM'] = np.ones(label_infos[1][1], dtype=np.float32)
  
  imgs = []
  carids = []
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carids.append(carid)
    carson = onecar['son']
    tmpath = carpath+'/'+carson
#    son = mx.image.imdecode(open(tmpath).read())
    son = cv2.imread(tmpath)
    imgs.append(son)

  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carid = int(onecar['id'])
   
    son = imgs[si] 
#    son = son.asnumpy()
#    print son.shape
    if rndcrop:
      son = get_rnd_crop(son)
#    print 0, tmpath, son0.shape, stdsize
    stdson = cv2.resize(son, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
    if rndcont:
      stdson = get_rnd_contrast(stdson)
    if rndnoise:
      stdson = get_rnd_noise(stdson)
    if normalize:
      stdson = get_normalization(stdson)
    if rndrotate:
      stdson = get_rnd_rotate(stdson)
    if rndhflip:
      stdson = get_rnd_hflip(stdson)
#    print carid, stdson
    datas['data'][si, 0] = stdson[:, :, 0]
    datas['data'][si, 1] = stdson[:, :, 1]
    datas['data'][si, 2] = stdson[:, :, 2]
    labels['proxy_yM'][si, carid] = 1
    labels['proxy_ZM'][si, carid] = 0
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)
  datas_nd = [mx.nd.array(datas['data'])]
  label_nd = [mx.nd.array(labels['proxy_yM']), mx.nd.array(labels['proxy_ZM'])]
  return datas_nd, label_nd, carids, batch_info


#format: path,imgname
def get_data_label_proxy_batch_mxnet_threads(data_infos, datas, label_infos, labels, datalist, batch_now, 
                   rndcrop=True, rndcont=False, rndnoise=False, rndrotate=True,
                   rndhflip=True, normalize=True):
#  print label_infos
  labelshape = label_infos[0][1]
  batchsize = labelshape[0]
  if (batch_now+1)*batchsize > len(datalist):
    return None
  
#  t0 = time.time()
  data_batch = []
  idlist = []
  batch_info = []
  for idx in xrange(batch_now*batchsize, (batch_now+1)*batchsize):
    data_batch.append(datalist[idx])
    idlist.append(idx)
  cars = []
  for idx, onedata in zip(idlist, data_batch):
    onecar = {}
    parts = onedata.split(',')
    onecar['path'] = parts[0]
    onecar['id'] = parts[-1] 
#    print onecar['id']
    onecar['son'] = parts[1]
    cars.append(onecar)
    carid = int(onecar['id'])
    oneinfo = '%s,%s,%s'%(parts[0], parts[1], parts[2])
    batch_info.append(oneinfo)

  stdsize = data_infos[0][1][2:]
  dataidx = 0
  labels['proxy_yM'][:] = 0
  labels['proxy_ZM'][:] = 1.0 
  
  tmpaths = []
  carids = []
  for si in xrange(batchsize):
    onecar = cars[si]
    carpath = onecar['path']
    carid = int(onecar['id'])
    carids.append(carid)
    carson = onecar['son']
    tmpath = carpath+'/'+carson
    tmpaths.append(tmpath)
 
#  t1 = time.time()
#  aug_data = aug_threads_c(tmpaths, data_infos[0][1])
  aug_data = datas['databuffer']
  aug_threads_c2(tmpaths, data_infos[0][1], aug_data)
#  t2 = time.time()

#  aug_data = aug_data.swapaxes(2, 3)
#  aug_data = aug_data.swapaxes(1, 2)
#  t3 = time.time()
  datas['data'][:] = aug_data

#  t4 = time.time()
  #ready same data
  for si in xrange(batchsize):
    onecar = cars[si]
    carid = int(onecar['id'])

    labels['proxy_yM'][si, carid] = 1
    labels['proxy_ZM'][si, carid] = 0
    if False:
      imgsave = (stdson*255).astype(np.uint8)
      cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), imgsave)
  datas_nd = [datas['data']]
  label_nd = [labels['proxy_yM'], labels['proxy_ZM']]
#  t5 = time.time()
#  print t5-t4, t4-t3, t3-t2, t2-t1, t1-t0

  return datas_nd, label_nd, carids, batch_info









