import logging
import numpy as np
import mxnet as mx

from DataIter import CarReID_Iter, CarReID_Test_Iter, CarReID_Feat_Iter
from Solver import CarReID_Solver
from Predictor import CarReID_Predictor, CarReID_Feature_Predictor, CarReID_Compare_Predictor
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
#  data_set_fn = '/home/mingzhang/data/car_ReID_for_zhangming/test/cam_1.200.list'
  data_set = CarReID_Test_Iter(['part2_data'], [data_shape], data_set_fn)

  batch_size = data_shape[0]
  reid_net = now_model.CreateModel_Color_Test(ctx, batch_size, data_shape[2:])
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params0/car_reid'
  predictor = CarReID_Predictor(param_prefix, reid_net, ctx, data_shape)

  print 'Testing...'
  resotre_whichone = 15 
  predictor.predict(data_query, data_set, whichone=resotre_whichone, logger=logger) 
  print 'over...'

  return 


def Do_Feature_Test(restore):
  print 'Extracting feature...'

  fdir = '/home/mingzhang/data/car_ReID_for_zhangming/test_train'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctx = mx.gpu(1)

  data_shape = (1, 3, 256, 256)
  data_query_fn = fdir + '/cam_0.list'
  data_query = CarReID_Test_Iter(['part1_data'], [data_shape], data_query_fn)
  data_set_fn = fdir + '/cam_1.list'
#  data_set_fn = fdir + '/cam_1.1w.list'
  data_set_fn = fdir + '/cam_1.2k.list'
#  data_set_fn = fdir + '/cam_1.200.list'
  data_set = CarReID_Test_Iter(['part1_data'], [data_shape], data_set_fn)

  batch_size = data_shape[0]
  reid_feature_net, _ = now_model.CreateModel_Color_Split_test()
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params0/car_reid'
  predictor_feature = CarReID_Feature_Predictor(param_prefix, reid_feature_net, ctx, data_shape)

  print 'Extracting feature...'
  resotre_whichone = restore 
  feat_savepath = fdir + '/cam_feat_0'
  predictor_feature.predict(data_query, feat_savepath, whichone=resotre_whichone, logger=logger) 
  feat_savepath = fdir + '/cam_feat_1'
  predictor_feature.predict(data_set, feat_savepath, whichone=resotre_whichone, logger=logger) 

  print 'over...'

  return


def Do_Compare_Test(restore):
  print 'Comparing feature...'

  fdir = '/home/mingzhang/data/car_ReID_for_zhangming/test_train'
  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctx = mx.gpu(1)

  data_shape = (1, 16384)
  data_query_fn = fdir+'/cam_feat_0.list'
  data_query = CarReID_Feat_Iter(['feature1_data'], [data_shape], data_query_fn)
  data_set_fn = fdir+'/cam_feat_1.list'
  data_set_fn = fdir+'/cam_feat_1.1w.list'
  data_set_fn = fdir+'/cam_feat_1.2k.list'
  data_set_fn = fdir+'/cam_feat_1.1k.list'
  data_set = CarReID_Feat_Iter(['feature1_data'], [data_shape], data_set_fn)

  batch_size = data_shape[0]
  _, reid_cmp_net = now_model.CreateModel_Color_Split_test()
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params0/car_reid'
  predictor_compare = CarReID_Compare_Predictor(param_prefix, reid_cmp_net, ctx, data_shape)

  print 'Comparing...'
  resotre_whichone = restore 
  predictor_compare.predict(data_query, data_set, whichone=resotre_whichone, logger=logger) 

  print 'over...'

  return



if __name__=='__main__':
#  Do_Test()
  restore_whichone = 23
  Do_Feature_Test(restore_whichone)
  Do_Compare_Test(restore_whichone)

