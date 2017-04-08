import os

def get_car_list(tgtdir, listfn):
  car_dirs = os.listdir(tgtdir)
  carlist = open(listfn, 'w')
  for car in car_dirs:
    car_dir = tgtdir + '/' + car
    car_files = os.listdir(car_dir)
    linetxt = car_dir
    for onecf in car_files:
      linetxt += ',' + onecf
    carlist.write(linetxt + '\n') 
  carlist.close()


def get_part_car_list(tgtdir, listfn, prenum):
  car_dirs = os.listdir(tgtdir)
  carlist = open(listfn, 'w')
  for car in car_dirs:
    if prenum>0 and int(car) >= prenum:
      continue
    car_dir = tgtdir + '/' + car
    car_files = os.listdir(car_dir)
    linetxt = car_dir
    for onecf in car_files:
      linetxt += ',' + onecf
    carlist.write(linetxt + '\n') 
  carlist.close()


def get_part_car_each_list(tgtdir, listfn, prenum):
  car_dirs = os.listdir(tgtdir)
  carlist = open(listfn, 'w')
  for car in car_dirs:
    if prenum>0 and int(car) >= prenum:
      continue
    car_dir = tgtdir + '/' + car
    car_files = os.listdir(car_dir)
    linetxt = car_dir
    for onecf in car_files:
      linetxttmp = linetxt + ',' + onecf
      carlist.write(linetxttmp + '\n') 
  carlist.close()



if __name__=='__main__':
  tgtdir = "/media/data1/mzhang/data/car_ReID_for_zhangming/data"
  tgtdir = "/home/mingzhang/data/car_ReID_for_zhangming/data"
  listfn = "data_each.500.list"
  get_part_car_each_list(tgtdir, listfn, 500)
#  get_part_car_list(tgtdir, listfn, 10)


