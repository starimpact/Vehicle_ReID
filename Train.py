import logging
import numpy as np
import mxnet as mx

from DataIter import CarReID_Iter, CarReID_Softmax_Iter
from Solver import CarReID_Solver, CarReID_Softmax_Solver
from MDL_PARAM import model2 as now_model
from MDL_PARAM import model1_softmax as softmax_model

def Do_Train():
  print 'Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctx = mx.gpu(3)

  num_epoch = 10000
  batch_size = 16
  data_shape = (batch_size, 3, 299, 299)
  label_shape = (batch_size, 1)
  data_train = CarReID_Iter(['part1_data', 'part2_data'], [data_shape, data_shape], ['label'], [label_shape])

  reid_net = now_model.CreateModel_Color(ctx, batch_size, data_shape[2:])
  
  dlr = 40000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]
  dlr_steps = [dlr*i for i in xrange(1, 80)]
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, 0.9)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params2/car_reid'
#  solver = CarReID_Solver(param_prefix, reid_net, ctx, data_shape, label_shape, num_epoch, 
#                    momentum=0.9, wd=0.0005, learning_rate=0.001, lr_scheduler=lr_scheduler)
  solver = CarReID_Solver(param_prefix, reid_net, ctx, data_shape, label_shape, num_epoch, 
                    opt_method='adam', wd=0.0005, learning_rate=0.0001, lr_scheduler=lr_scheduler)


  print 'fitting...'
  resotre_whichone = None
  solver.fit(data_train, showperiod=100, whichone=resotre_whichone, logger=logger) 
  print 'over...'

  return 

def Do_Softmax_Train():
  print 'Softmax Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctx = mx.gpu(3)

  num_epoch = 10000
  batch_size = 16
  data_shape = (batch_size, 3, 299, 299)
  data_shape = (batch_size, 3, 256, 256)
  label_shape = (batch_size, )
  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.list' #43928 calss number.
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.10.list'
  data_train = CarReID_Softmax_Iter(['data'], [data_shape], ['label'], [label_shape], datafn)
  clsnum=43928
  reid_net = softmax_model.CreateModel_Color(ctx, batch_size, data_shape[2:], clsnum)
  
  dlr = 200000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]
  dlr_steps = [dlr*i for i in xrange(1, 80)]
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, 0.9)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params1_softmax/car_reid'
  solver = CarReID_Softmax_Solver(param_prefix, reid_net, ctx, data_shape, label_shape, num_epoch, 
                    momentum=0.9, wd=0.0005, learning_rate=0.0003, lr_scheduler=lr_scheduler)
#  solver = CarReID_Softmax_Solver(param_prefix, reid_net, ctx, data_shape, label_shape, num_epoch, 
#                    opt_method='rmsprop', wd=0.0005, learning_rate=0.01, lr_scheduler=lr_scheduler)


  print 'fitting...'
  resotre_whichone = 4
  solver.fit(data_train, showperiod=100, whichone=resotre_whichone, logger=logger) 
  print 'over...'

  return 




if __name__=='__main__':
#  Do_Train()
  Do_Softmax_Train()


