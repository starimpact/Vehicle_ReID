import logging
import numpy as np
import mxnet as mx

from DataIter import CarReID_Iter
from Solver import CarReID_Solver
from MDL_PARAM import model0 as now_model

def Do_Train():
  print 'Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctx = mx.gpu(3)

  num_epoch = 10000
  batch_size = 64
  data_shape = (batch_size, 3, 128, 128)
  label_shape = (batch_size, 1)
  data_train = CarReID_Iter(['part1_data', 'part2_data'], [data_shape, data_shape], ['label'], [label_shape])

  reid_net = now_model.CreateModel_Color(ctx, batch_size, data_shape[2:])
  
  solver = CarReID_Solver('MDL_PARAM/params0/car_reid', reid_net, ctx, data_shape, label_shape, num_epoch, momentum=0.9, wd=0.0005,
                    learning_rate=0.01, lr_scheduler=mx.lr_scheduler.FactorScheduler(1000, 0.1))

  print 'fitting...'
  solver.fit(data_train, showperiod=500, logger=logger) 
  print 'over...'

  return 

if __name__=='__main__':
  Do_Train()


