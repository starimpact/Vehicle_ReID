import numpy as np
import os
import cPickle

def mysort(car0, car1):
  return -cmp(car0['score'], car1['score'])

def top20(fn):
  carlist = []
  for line in file(fn):
    line = line.replace('\n', '')
    parts = line.split(',')
    car = {}
    car['id'] = parts[0]
    car['name'] = parts[1]
    car['score'] = float(parts[2])
    carlist.append(car)
#  print len(carlist)
  carlist.sort(mysort)
#  print carlist
#  exit()
  top_20 = carlist[:20]
  return top_20

def get_all_top(folder):
  qcars_list = []
  for qcar in os.listdir(folder):
    fname = qcar.split('.')[0]
    if qcar.endswith('list') is False:
      continue 
    parts = fname.split('_')
#    print qcar, parts
    qcarinfo = {}
    qcarinfo['id'] = parts[1]
    qcarinfo['name'] = parts[2]
    top_20 = top20(folder+'/'+qcar)
    qcarinfo['data'] = top_20
    print qcarinfo
    qcars_list.append(qcarinfo) 
  return qcars_list


def get_rightrate(qcar_list):
  topnum_list = [1, 5, 10, 15, 20]
  topsize = len(topnum_list)
  toprate_num = np.zeros(topsize) 
  for qcar in qcar_list:
    qid = qcar['id']
    qname = qcar['name']
    idx = 0
    useds = np.zeros(topsize, np.bool)
    for scar in qcar['data']:
      sid = scar['id']
      sname = scar['name']
      if sid==qid:
        for n, topid in enumerate(topnum_list):
          if idx < topid: 
            useds[n:] = True
            break
      idx+=1
    toprate_num[useds]+=1 
  print toprate_num 
    


if __name__=='__main__':
  if False:
    folder = 'Result'
    qcar_list = get_all_top(folder)
    cPickle.dump(qcar_list, open('top20.bin', 'wb'))
  if True:
    qcar_list = cPickle.load(open('top20.bin', 'rb'))
    get_rightrate(qcar_list)






