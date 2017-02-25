import numpy as np
import mxnet as mx

from DataIter import CarReID_Iter
from Solver import CarReID_Solver
from MDL_PARAM import model0 as now_model

def Do_Train():
  print 'Training...'
  
  ctx = mx.cpu()

  num_epoch = 20
  batch_size = 32
  data_shape = (batch_size, 3, 128, 128)
  label_shape = (batch_size, 1)
  data_train = CarReID_Iter(['part1_data', 'part2_data'], [data_shape, data_shape], ['label'], [label_shape])

  reid_net = now_model.CreateModel_Color(ctx, batch_size, data_shape[2:])
  
  solver = CarReID_Solver('MDL_PARAM/params0/car_reid', reid_net, ctx, data_shape, label_shape, num_epoch, momentum=0.9, wd=0.0005,
                    learning_rate=0.01, lr_scheduler=mx.lr_scheduler.FactorScheduler(100, 0.1))

  print 'fitting...'
  solver.fit(data_train) 
  print 'over...'

  return 

if __name__=='__main__':
  Do_Train()


