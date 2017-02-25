import numpy as np
import cv2

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
  datas['part1_data'] = np.zeros(data_infos[0][1])
  datas['part2_data'] = np.zeros(data_infos[1][1])
  labels['label'] = np.zeros(label_infos[0][1])
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
    tmpath = carpath+'/'+carsons[rndidx[1]]
    son1 = cv2.imread(tmpath)
#    print 1, tmpath, son1.shape, stdsize
    stdson1 = cv2.resize(son1, (stdsize[1], stdsize[0]))
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
    son1 = cv2.imread(car1['path']+'/'+car1['sons'][np.random.randint(0, car1len)])
    stdson1 = cv2.resize(son1, (stdsize[1], stdsize[0]))
    datas['part1_data'][dataidx, 0] = stdson0[:, :, 0]
    datas['part1_data'][dataidx, 1] = stdson0[:, :, 1]
    datas['part1_data'][dataidx, 2] = stdson0[:, :, 2]
    datas['part2_data'][dataidx, 0] = stdson1[:, :, 0]
    datas['part2_data'][dataidx, 1] = stdson1[:, :, 1]
    datas['part2_data'][dataidx, 2] = stdson1[:, :, 2]
    labels['label'][dataidx] = -1
    dataidx += 1

  return datas, labels


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



