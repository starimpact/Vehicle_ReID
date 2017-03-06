import numpy as np
import cv2
import cPickle

datafn = '/media/data1/mzhang/data/car_ReID_for_zhangming/data/data.list'

def get_datalist(datafn):
  datafile = open(datafn, 'r')
  datalist = datafile.readlines()
  datalen = len(datalist)
  for di in xrange(datalen):
    datalist[di] = datalist[di].replace('\n', '')
    
  datafile.close()
  
  return datalist


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


def get_data_label_test(data_shape, datalist, which_car):
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


def get_data_label(data_infos, label_infos, datalist, data_rndidx, batch_now):
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
#    print 0, tmpath, son0.shape, stdsize
    stdson = cv2.resize(son, (stdsize[1], stdsize[0]))
    stdson = stdson.astype(np.float32) / 255.0
#    print carid, stdson
    datas['data'][si, 0] = stdson[:, :, 0]
    datas['data'][si, 1] = stdson[:, :, 1]
    datas['data'][si, 2] = stdson[:, :, 2]
    labels['label'][si] = carid
#    cv2.imwrite('tmpimg/stdson%d.jpg'%(int(carid)), stdson)

  return datas, labels




