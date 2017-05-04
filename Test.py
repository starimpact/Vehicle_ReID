import logging
import numpy as np
import mxnet as mx
import cPickle
import time
from ctypes import *

 
import DataGenerator as dg
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


def do_predict_feature(predict_model, data_iter, savefolder, nn):
  data_iter.reset()
  needbnum = int(np.ceil(nn*1.0/data_iter.batch_size))
  for data in data_iter:
    print 'feature extracting...%.2f%%(%d/%d, %d), to %s'%(data_iter.cur_batch*100.0/data_iter.num_batches, data_iter.cur_batch, data_iter.num_batches, needbnum, savefolder)
    predict_model.forward(data)
    feats = predict_model.get_outputs()[0].asnumpy()
    labels = data.label[0].asnumpy()
    cPickle.dump([labels, feats, data_iter.paths], open('%s/%d.feat'%(savefolder, data_iter.cur_batch), 'wb'))
    if needbnum>0 and data_iter.cur_batch>=needbnum:
      break
  pass



def create_compare_feature_model(ctxs, provide_data):
  imgnum, featdim = provide_data[1][1]
  _, reid_feature_net = now_model.CreateModel_Color_Split_test()
  data_names = []
  for dn in provide_data:
    data_names.append(dn[0])
  reid_model = mx.mod.Module(context=ctxs, symbol=reid_feature_net, data_names=data_names)
  reid_model.bind(data_shapes=provide_data, for_training=False)
  reid_model.init_params()
  return reid_model


def do_compare_feature(predict_model, bsz, query_list, distractor_list, savefolder):
  for qfn in query_list:
    labels_q, datas_q, paths = cPickle.load(open(qfn, 'rb'))
    qlen = np.sum(labels_q[:, 0]>-1)
    for qi in xrange(qlen):
      data1 = datas_q[qi:qi+1]
      rep_data1 = data1.repeat(bsz, axis=0)
      nd_data1 = mx.nd.array(rep_data1)
      id1 = str(labels_q[qi, 0])
      type1 = labels_q[qi, 1]
      path1 = paths[qi]
      name1 = path1.split('/')[-1]
      cmpfile = open(savefolder+'/cmp=%s=%s.list'%(id1, name1), 'w')
      t0 = time.time()
      for dfn in distractor_list:
        labels_d, datas_d, paths_d = cPickle.load(open(dfn, 'rb'))
        nd_data2 = mx.nd.array(datas_d)
        data = mx.io.DataBatch([nd_data1, nd_data2], [])
        predict_model.forward(data)
        cmp_scores = predict_model.get_outputs()[0].asnumpy()
        cmp_scores = np.sum(cmp_scores, axis=1)
        dlen = np.sum(labels_d[:, 0]>-1)
        
        writestrs = ''
        for bi in xrange(dlen):
          id2 = labels_d[bi, 0]
          path2 = paths_d[bi]
          name2 = path2.split('/')[-1]
          cmp_score = cmp_scores[bi]
          writestrs += '%s,%s,%f\n'%(id2, name2, cmp_score)
        cmpfile.write(writestrs)
        cmpfile.flush()
      cmpfile.close()
      t1 = time.time()
      print '%s, %d/%d->time cost:%.3f s'%(id1, qi, qlen, (t1-t0))
  pass


rank_func = CDLL('./ranker/libranker.so')

def init_ranker_c(database, indexes):
  dbsize, featdim = database.shape
  rank_func.init_ranker(database.ctypes.data_as(POINTER(c_float)), indexes.ctypes.data_as(POINTER(c_int)), dbsize, featdim)
  pass 


def do_ranker_c(query, topNIdxes, topNScores):
  featdim = query.shape[0]
  topN = topNIdxes.shape[0]
  topNIdxes[:] = 0
  topNScores[:] = 0
  rank_func.do_ranker(query.ctypes.data_as(POINTER(c_float)), featdim, 
                      topNIdxes.ctypes.data_as(POINTER(c_int)),
                      topNScores.ctypes.data_as(POINTER(c_float)), topN)
  pass
  

def do_fill_dataset_c(distractor_list, savefolder=''):
  print 'loading whole distractor set...'
  dbsize = 0
  featdim = 0
  for dfn in distractor_list:
    labels_d, datas_d, paths_d = cPickle.load(open(dfn, 'rb'))
    realnum = np.sum(labels_d[:, 0] > -1)
    dbsize += realnum
    featdim = datas_d.shape[1]
  datas_dall = np.zeros((dbsize, featdim), dtype=np.float32)
  labels_dall = np.zeros((dbsize, 2), dtype=np.int32)
  print 'data size:%d, feat dim:%d'%(dbsize, featdim)
  paths_dall = []
  dbposnow = 0
  for dfn in distractor_list:
    labels_d, datas_d, paths_d = cPickle.load(open(dfn, 'rb'))
    realnum = np.sum(labels_d[:, 0] > -1)
    datas_dall[dbposnow:dbposnow+realnum] = datas_d[:realnum]
    labels_dall[dbposnow:dbposnow+realnum] = labels_d[:realnum]
    paths_dall += paths_d[:realnum]
    dbposnow += realnum
  assert(dbposnow==dbsize)
  indexes = np.asarray(range(dbsize), dtype=np.int32)
  init_ranker_c(datas_dall, indexes)
  
  return labels_dall, paths_dall 

def do_compare_feature_c(labels_dall, paths_dall, query_list, qtype='', savefolder=''):
  print 'quering...'
  qcars_list = []
  topN = 100+1
  topNIdxs = np.zeros((topN,), dtype=np.int32)
  topNScores = np.zeros((topN,), dtype=np.float32)
  samenum_q = np.zeros((topN, 2), dtype=np.int32)
  allnum_q = np.zeros((topN, 2), dtype=np.int32)
  for qfn in query_list:
    print qfn
    labels_q, datas_q, paths = cPickle.load(open(qfn, 'rb'))
    qlen = np.sum(labels_q[:, 0]>-1)
    t0 = time.time()
    for qi in xrange(qlen):
      data1 = datas_q[qi]
      id1 = labels_q[qi, 0]
      tp1 = labels_q[qi, 1]
      qcarinfo = {}
      qcarinfo['id'] = id1
      qcarinfo['path'] = paths[qi]
      do_ranker_c(data1, topNIdxs, topNScores) 
#      if id1==2293:
#        print qi+1, topNIdxs, topNScores
      allid2 = labels_dall[topNIdxs, 0]
      alltp2 = labels_dall[topNIdxs, 1]
      carlist = []
      for idx in xrange(topN):
        car = {}
        car['id'] = allid2[idx]
        car['path'] = paths_dall[topNIdxs[idx]]
        car['score'] = topNScores[idx]
        carlist.append(car)
        if tp1==0:
          if id1 in allid2[:idx+1]:
            samenum_q[idx, 0] += 1
          allnum_q[idx, 0] += 1  
        else:
          if id1 in allid2[:idx+1]:
            samenum_q[idx, 1] += 1
          allnum_q[idx, 1] += 1
      qcarinfo['data'] = carlist
      qcars_list.append(qcarinfo)
    t1 = time.time()
  topName = 'topN_%s.bin'%(qtype)
  cPickle.dump(qcars_list, open(topName, 'wb'))
  print 'saved topN into', topName
  ratios = samenum_q / (allnum_q + 10**-16)
  needN = np.asarray(range(0, topN, 10))
  print 'topN     :', needN+1
  print 'has plate:', ratios[needN][:, 0].T, 'num:%d'%allnum_q[0, 0]
  print 'no  plate:', ratios[needN][:, 1].T, 'num:%d'%allnum_q[0, 1]
  print 'time cost:%.3f s'%(t1-t0)
  
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
#  neednums = [800, 0]
#  data_query_fn = [fdir+'/cam_each_0.list', fdir+'/cam_each_1.list']
#  save_folder_fn = [fdir+'/cam_feat_quick_0', fdir+'/cam_feat_quick_1'] 

  fdir = '/mnt/ssd2/minzhang/ReID_BigBenchMark/mingzhang'
  neednums = [1600, 1600, 0, 0]
  data_query_fn = [fdir+'/front_image_list_query.list', 
                   fdir+'/back_image_list_query.list',
                   fdir+'/front_image_list_distractor.list',
                   fdir+'/back_image_list_distractor.list']
  save_folder_fn = [fdir+'/front_image_query', 
                    fdir+'/back_image_query',
                    fdir+'/front_image_distractor',
                    fdir+'/back_image_distractor'] 

  t0 = time.time()
  for nn, d1, d2 in zip(neednums, data_query_fn, save_folder_fn):
    data_query = CarReID_TestQuick_Iter(['part1_data'], [data_shape], ['id'], [label_shape], [d1])
    do_predict_feature(feature_model, data_query, d2, nn)
  t1 = time.time()
  print 'extracted all features costs', t1-t0
 
  print 'over...'

  return



def Do_Feature_Compare_Fast():
  print 'comparing feature...'
  ctxs = [mx.gpu(0)]
  bsz = 800
  data_shape1 = (bsz, 128) #model2_proxy_nca
  data_shape2 = (bsz, 128) #model2_proxy_nca
  provide_data = [['feature1_data', data_shape1], ['feature2_data', data_shape2]]
 
#  fdir = '/mnt/ssd2/minzhang/Re-ID_select'
#  querylist_fn = [fdir+'/cam_feat_quick_0.list'] 
#  distractorlist_fn = [fdir+'/cam_feat_quick_1.list'] 
  savefolder = 'Result'

  fdir = '/mnt/ssd2/minzhang/ReID_BigBenchMark/mingzhang'
  querylist_fn = [fdir+'/front_image_query.list', 
                  fdir+'/back_image_query.list'] 
  distractorlist_fn = [fdir+'/front_image_distractor.list',
                       fdir+'/back_image_distractor.list']
  qtypes = ['front', 'back']

  query_lists = []
  for qfn in querylist_fn:
    query_list_one = dg.get_datalist2([qfn]) 
    query_lists.append(query_list_one)
  distractor_list = dg.get_datalist2(distractorlist_fn) 
 
  if 0:
    compare_model = create_compare_feature_model(ctxs, provide_data)
    do_compare_feature(compare_model, bsz, query_list, distractor_list, savefolder) 
  else:
    labels_dall, paths_dall = do_fill_dataset_c(distractor_list)
    for qtype, qlist in zip(qtypes, query_lists):
      do_compare_feature_c(labels_dall, paths_dall, qlist, qtype) 

  pass

if __name__=='__main__':
#  Do_Test()
  restore_whichone = 3
  ctx = mx.gpu(0)
#  Do_Softmax_Test_Acc(ctx, restore_whichone)
#  Do_Feature_Test(restore_whichone, ctx)
#  Do_Compare_Test(restore_whichone, ctx)
#############
  if 0:
    Do_Feature_Test_Fast(restore_whichone)
  else:
    Do_Feature_Compare_Fast()


