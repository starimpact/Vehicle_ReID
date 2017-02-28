import logging
import numpy as np
import mxnet as mx

from DataIter import CarReID_Iter, CarReID_Test_Iter
from Solver import CarReID_Solver
from Predictor import CarReID_Predictor
from MDL_PARAM import model0 as now_model

def Do_Test():
  print 'Testing...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctx = mx.gpu(1)

  data_shape = (1, 3, 256, 256)
  data_query_fn = '/home/mingzhang/data/car_ReID_for_zhangming/test/cam_0.list'
  data_query = CarReID_Test_Iter(['part1_data'], [data_shape], data_query_fn)
 # data_set_fn = '/home/mingzhang/data/car_ReID_for_zhangming/test/cam_1.list'
#  data_set_fn = '/home/mingzhang/data/car_ReID_for_zhangming/test/cam_1.1w.list'
  data_set_fn = '/home/mingzhang/data/car_ReID_for_zhangming/test/cam_1.1k.list'
  data_set = CarReID_Test_Iter(['part2_data'], [data_shape], data_set_fn)
  data_set = CarReID_Test_Iter(['part2_data'], [data_shape], data_set_fn)

  batch_size = data_shape[0]
  reid_net = now_model.CreateModel_Color_Test(ctx, batch_size, data_shape[2:])
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params0/car_reid'
  predictor = CarReID_Predictor(param_prefix, reid_net, ctx, data_shape)

  print 'Testing...'
  resotre_whichone = 43 
  predictor.predict(data_query, data_set, whichone=resotre_whichone, logger=logger) 
  print 'over...'

  return 

if __name__=='__main__':
  Do_Test()


