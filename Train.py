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
  data_shape = (batch_size, 3, 256, 256)
  label_shape = (batch_size, 1)
  data_train = CarReID_Iter(['part1_data', 'part2_data'], [data_shape, data_shape], ['label'], [label_shape])

  reid_net = now_model.CreateModel_Color(ctx, batch_size, data_shape[2:])
  
  dlr = 20000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]
  dlr_steps = [dlr*i for i in xrange(1, 80)]
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, 0.9)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params0/car_reid'
  solver = CarReID_Solver(param_prefix, reid_net, ctx, data_shape, label_shape, num_epoch, 
                    momentum=0.9, wd=0.0005, learning_rate=0.0001, lr_scheduler=lr_scheduler)

  print 'fitting...'
  resotre_whichone = 1
  solver.fit(data_train, showperiod=100, whichone=resotre_whichone, logger=logger) 
  print 'over...'

  return 

if __name__=='__main__':
  Do_Train()


