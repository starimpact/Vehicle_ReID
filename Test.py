import logging
import numpy as np
import mxnet as mx
import cPickle
import time
 
from DataIter import CarReID_Iter, CarReID_Test_Iter, CarReID_Feat_Query_Iter, CarReID_Feat_Iter, CarReID_Softmax_Iter
from DataIter import CarReID_TestQuick_Iter
from Solver import CarReID_Solver
from Predictor import CarReID_Predictor, CarReID_Feature_Predictor, CarReID_Compare_Predictor, CarReID_Softmax_Predictor
#from MDL_PARAM import model2_softmax as now_model
#from MDL_PARAM import model2_proxy_nca as now_model
from MDL_PARAM import model3_proxy_nca as now_model

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


def Do_Feature_Test(restore, ctx=mx.cpu()):
  print 'Extracting feature...'

  fdir = '/home/mingzhang/data/car_ReID_for_zhangming/test_train'
  fdir = '/home/mingzhang/data/car_ReID_for_zhangming/test'
  fdir = '/mnt/ssd2/minzhang/Re-ID_select'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  

  data_shape = (1, 3, 256, 256)
  data_shape = (1, 3, 299, 299)
  data_query_fn = fdir + '/cam_0.list'
  data_query = CarReID_Test_Iter(['part1_data'], [data_shape], data_query_fn, normalize=True)
  data_set_fn = fdir + '/cam_1.list'
#  data_set_fn = fdir + '/cam_1.1w.list'
#  data_set_fn = fdir + '/cam_1.2k.list'
#  data_set_fn = fdir + '/cam_1.200.list'
  data_set = CarReID_Test_Iter(['part1_data'], [data_shape], data_set_fn, normalize=True)

  batch_size = data_shape[0]
#  reid_feature_net, _ = now_model.CreateModel_Color_Split_test()
  reid_feature_net, _ = now_model.CreateModel_Color_Split_test2()
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
#  param_prefix = 'MDL_PARAM/params2_softmax/car_reid'
  param_prefix = 'MDL_PARAM/params2_proxy_nca_combine/car_reid'
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'
  param_prefix = 'MDL_PARAM/params3_proxy_nca/car_reid'
  predictor_feature = CarReID_Feature_Predictor(param_prefix, reid_feature_net, ctx, data_shape)

  print 'Extracting feature...'
  resotre_whichone = restore 
  feat_savepath = fdir + '/cam_feat_0'
  predictor_feature.predict(data_query, feat_savepath, whichone=resotre_whichone, logger=logger) 
  feat_savepath = fdir + '/cam_feat_1'
  predictor_feature.predict(data_set, feat_savepath, whichone=resotre_whichone, logger=logger) 

  print 'over...'

  return


def Do_Compare_Test(restore, ctx=mx.cpu()):
  print 'Comparing feature...'

  fdir = '/home/mingzhang/data/car_ReID_for_zhangming/test_train'
  fdir = '/home/mingzhang/data/car_ReID_for_zhangming/test'
  fdir = '/mnt/ssd2/minzhang/Re-ID_select'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  

  data_shape = (1000, 16384) #model0
  data_shape = (1000, 1536) #model2
  data_shape = (1000, 512) #model1_softmax
  data_shape = (1000, 256) #model1_softmax2, model2_softmax
  data_shape1 = (1, 128) #model2_proxy_nca
  data_shape2 = (10000, 128) #model2_proxy_nca
  data_query_fn = fdir+'/cam_feat_0.list'
  data_query = CarReID_Feat_Query_Iter(['feature1_data'], [data_shape1], data_query_fn)
  data_set_fn = fdir+'/cam_feat_1.list'
#  data_set_fn = fdir+'/cam_feat_1.1w.list'
#  data_set_fn = fdir+'/cam_feat_1.2k.list'
#  data_set_fn = fdir+'/cam_feat_1.1k.list'
  data_set = CarReID_Feat_Iter(['feature2_data'], [data_shape2], data_set_fn)

  batch_size = data_shape[0]
#  _, reid_cmp_net = now_model.CreateModel_Color_Split_test()
  _, reid_cmp_net = now_model.CreateModel_Color_Split_test2(data_shape2[0], data_shape2[1])
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
#  param_prefix = 'MDL_PARAM/params2_softmax/car_reid'
  param_prefix = 'MDL_PARAM/params2_proxy_nca_combine/car_reid'
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'
  param_prefix = 'MDL_PARAM/params3_proxy_nca/car_reid'
  predictor_compare = CarReID_Compare_Predictor(param_prefix, reid_cmp_net, ctx, data_shape2)

  print 'Comparing...'
  resotre_whichone = restore 
  predictor_compare.predict(data_query, data_set, whichone=resotre_whichone, logger=logger) 

  print 'over...'

  return


def Do_Softmax_Test_Acc(ctx, resotre_whichone):
  print 'Softmax test accuracy...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  batch_size = 10
  data_shape = (batch_size, 3, 256, 256)
  label_shape = (batch_size, )
  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.list' #43928 calss number.
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.10.list'
  data_train = CarReID_Softmax_Iter(['data'], [data_shape], ['label'], [label_shape], datafn)
  clsnum=43928
  reid_net = now_model.CreateModel_Color(ctx, batch_size, data_shape[2:], clsnum)
  
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params2_softmax/car_reid'
  predictor = CarReID_Softmax_Predictor(param_prefix, reid_net, ctx, data_shape)


  print 'predicting...'
  showp = 100
  predictor.predict(data_train, showperiod=showp, whichone=resotre_whichone, logger=logger) 
  print 'over...'

  return 


def load_checkpoint(model, prefix, epoch):
    param_name = '%s-%04d.params' % (prefix, epoch)
    save_dict = mx.nd.load(param_name)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            arg_params[name] = value
        elif arg_type == 'aux':
            aux_params[name] = value
        else:
            raise ValueError("Invalid param file " + fname)
    model.set_params(arg_params, aux_params, allow_missing=True)
    arg_params, aux_params = model.get_params()
    logging.info('Load checkpoint from \"%s\"', param_name)
    return arg_params, aux_params


def create_predict_feature_model(ctxs, provide_data, param_prefix, load_paramidx):
  reid_feature_net, _ = now_model.CreateModel_Color_Split_test()
  data_names = []
  for dn in provide_data:
    data_names.append(dn[0])
  reid_model = mx.mod.Module(context=ctxs, symbol=reid_feature_net, data_names=data_names)
  reid_model.bind(data_shapes=provide_data, for_training=False)
  arg_params, aux_params = load_checkpoint(reid_model, param_prefix, load_paramidx)

  return reid_model


def do_predict_feature(predict_model, data_iter, savefolder):
  data_iter.reset()
  for data in data_iter:
    print 'feature extracting...%.2f%%(%d/%d), to %s'%(data_iter.cur_batch*100.0/data_iter.num_batches, data_iter.cur_batch, data_iter.num_batches, savefolder)
    predict_model.forward(data)
    feats = predict_model.get_outputs()[0].asnumpy()
    labels = data.label[0].asnumpy()
    cPickle.dump([labels, feats, data_iter.paths], open('%s/%d.feat'%(savefolder, data_iter.cur_batch), 'wb'))
  pass


def Do_Feature_Test_Fast(load_paramidx):
  print 'Extracting feature Fast...'


  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  batchsize = 200 * len(ctxs)
  data_shape = (batchsize, 3, 299, 299)
  label_shape = (batchsize, 2)

  param_prefix = 'MDL_PARAM/params2_proxy_nca_combine/car_reid'
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'
  param_prefix = 'MDL_PARAM/params3_proxy_nca/car_reid'
  feature_model = create_predict_feature_model(ctxs, [['part1_data', data_shape]], param_prefix, load_paramidx)

  fdir = '/mnt/ssd2/minzhang/Re-ID_select'
  data_query_fn = [fdir+'/cam_each_0.list', fdir+'/cam_each_1.list']
  save_folder_fn = [fdir+'/cam_feat_quick_0', fdir+'/cam_feat_quick_1'] 

#  fdir = '/mnt/ssd2/minzhang/ReID_BigBenchMark/mingzhang'
#  data_query_fn = [fdir+'/front_image_list_query.list', 
#                   fdir+'/back_image_list_query.list',
#                   fdir+'/front_image_list_distractor.list',
#                   fdir+'/back_image_list_distractor.list']
#  save_folder_fn = [fdir+'/front_image_query', 
#                    fdir+'/back_image_query',
#                    fdir+'/front_image_distractor',
#                    fdir+'/back_image_distractor'] 

  t0 = time.time()
  for d1, d2 in zip(data_query_fn, save_folder_fn):
    data_query = CarReID_TestQuick_Iter(['part1_data'], [data_shape], ['id'], [label_shape], [d1])
    do_predict_feature(feature_model, data_query, d2)
    exit()
  t1 = time.time()
  print 'extracted all features costs', t1-t0
 
  print 'over...'

  return



if __name__=='__main__':
#  Do_Test()
  restore_whichone = 3
  ctx = mx.gpu(0)
#  Do_Softmax_Test_Acc(ctx, restore_whichone)
#  Do_Feature_Test(restore_whichone, ctx)
#  Do_Compare_Test(restore_whichone, ctx)
  Do_Feature_Test_Fast(restore_whichone)


