import logging
import numpy as np
import mxnet as mx
from mxnet import metric

from DataIter import CarReID_Proxy_Batch_Mxnet_Iter
from Solver import CarReID_Solver, CarReID_Softmax_Solver, CarReID_Proxy_Solver
from MDL_PARAM import model2 as now_model
from MDL_PARAM import model2_proxy_nca as proxy_nca_model



class Proxy_Metric(metric.EvalMetric):
  def __init__(self, saveperiod=1):
    super(Proxy_Metric, self).__init__('proxy_metric')
    print "hello metric init..."
    self.num_inst = 0
    self.sum_metric = 0.0
    self.p_inst = 0
    self.saveperiod=saveperiod

#  def reset(self):
#    pass

  def update(self, labels, preds):
#    print '=========%d========='%(self.p_inst)
    self.p_inst += 1
    if self.p_inst%self.saveperiod==0:
      self.num_inst += 1
      loss = preds[0].asnumpy().mean()
#      print 'metric', loss
      self.sum_metric += loss
    

def Do_Proxy_NCA_Train2():
  print 'Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1)]
  ctxs = [mx.gpu(1)]
  
  devicenum = len(ctxs) 

  num_epoch = 10000
  batch_size = 32*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  proxy_batch = 100
  proxy_num = proxy_batch
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_Z_shape = (proxy_num, featdim)
  proxy_ZM_shape = (batch_size, proxy_num)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.list' #43928 calss number.
  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.500.list'
#  data_train = CarReID_Proxy2_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, bucket_key)
  data_train = CarReID_Proxy_Batch_Mxnet_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, proxy_batch)
  
  dlr = 200000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]

  lr_start = (10**-1)
  lr_min = 10**-6
  lr_reduce = 0.9
  lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
  lr_stepnum = np.int(np.ceil(lr_stepnum))
  dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
  print 'lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(lr_start, lr_min, lr_reduce, lr_stepnum)
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'

  reid_net = proxy_nca_model.CreateModel_Color2(None, bsz_per_device, proxy_num, data_shape[2:])
  reid_net_p = proxy_nca_model.CreateModel_Color_predict()


  reid_model = mx.mod.Module(context=ctxs, symbol=reid_net, 
                             label_names=['proxy_yM', 'proxy_ZM'])
  reid_model_P = mx.mod.Module(context=mx.gpu(0), symbol=reid_net_p)
#
  reid_model_P.bind(data_shapes=data_train.provide_data, for_training=False)


  optimizer_params={'learning_rate':lr_start,
                    'momentum':0.9,
                    'wd':0.0005,
                    'lr_scheduler':lr_scheduler,
                    'clip_gradient':None,
                    'rescale_grad': 1.0/batch_size}

  proxy_metric = Proxy_Metric()

  if False:
    fn = param_prefix + '_0_' + '.bin'
    reid_model.bind(data_shapes=data_train.provide_data, 
                    label_shapes=data_train.provide_label)
    reid_model.load_params(fn)
    print 'loaded parameters from', fn


  def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)

  mon = mx.mon.Monitor(1, norm_stat, 
                       pattern='.*part1_fc1.*|.*proxy_Z_weight.*')

  def batch_end_call(*args, **kwargs):
  #  print eval_metric.loss_list
    epoch = args[0].epoch
    nbatch = args[0].nbatch + 1
    eval_metric = args[0].eval_metric
    data_batch = args[0].locals['data_batch']  
    if nbatch%show_period==0:
       fn = param_prefix + '_' + str(epoch%4) + '_' + '.bin'
       reid_model.save_params(fn)
       print 'saved parameters into', fn
       eval_metric.reset()
       if False:
         args, auxs = reid_model.get_params()
         proxy_Z = args['proxy_Z_weight'].asnumpy()
         print 'z:', np.abs(proxy_Z).mean()
         reid_model_P.set_params(args, auxs)
         reid_model_P.forward(data_batch)
         outs = reid_model_P.get_outputs()
         val = outs[1].asnumpy()
         val2 = outs[2].asnumpy()
         val3 = outs[3].asnumpy()
         val4 = outs[4].asnumpy()
         print 'fc1:', np.abs(val).mean(), 'z:', val2.mean(), 'y:', val3, 'one_proxy:', np.abs(val4).mean(), val2[val2>val1]
#       args, auxs = reid_model.get_params()
#       print np.mean(args['proxy_Z_weight'].asnumpy())

  reid_model_P.init_params()
  arg_params_p, aux_params_p = reid_model_P.get_params()
  def epoch_end_call(epoch, symbol, arg_params, aux_params):
    print len(arg_params), len(arg_params_p)
    reid_model_P.set_params(arg_params, aux_params)
    carnum = data_train.do_reset()

    print 'hello end epoch...ready next proxy batch data and init the proxy_Z_weight...'

    proxy_Z_weight = arg_params['proxy_Z_weight']
    proxy_num, featdim = proxy_Z_weight.shape
    proxy_Zfeat = np.ones((proxy_num, featdim), dtype=np.float32)*10**-5
    proxy_Znum = np.zeros((proxy_num, 1), dtype=np.int32)
    for di, data in enumerate(data_train):
      output = reid_model_P.forward(data, is_train=False)
      output = reid_model_P.get_outputs()[0].asnumpy()
      batch_carids = data_train.batch_carids
      for ri in xrange(data_train.batch_size):
        carid = batch_carids[ri]
        proxy_Zfeat[carid] += output[ri]
        proxy_Znum[carid] += 1
    print proxy_Znum.T
    print proxy_Zfeat
    proxy_Znum[proxy_Znum==0] = 1
    proxy_Zfeat /= proxy_Znum
    print np.mean(proxy_Zfeat)
    proxy_Z_weight[:] = proxy_Zfeat 
    reid_model.set_params(arg_params, aux_params)
    pass

  batch_end_calls = [batch_end_call, mx.callback.Speedometer(batch_size, show_period/10)]
  epoch_all_calls = [epoch_end_call]
  reid_model.fit(train_data=data_train, eval_metric=proxy_metric,
                 optimizer='sgd',
                 optimizer_params=optimizer_params, 
                 initializer=mx.init.Normal(),
                 begin_epoch=0, num_epoch=num_epoch, 
                 eval_end_callback=None,
                 kvstore=None,# monitor=mon,
                 batch_end_callback=batch_end_calls,
                 epoch_end_callback=epoch_all_calls) 


  return 


def Do_Proxy_NCA_Train3():
  print 'Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1)]
#  ctxs = [mx.gpu(1)]
  
  devicenum = len(ctxs) 

  num_epoch = 10000
  batch_size = 32*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  proxy_num = 548597
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_Z_shape = (proxy_num, featdim)
  proxy_ZM_shape = (batch_size, proxy_num)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
  datapath = '/mnt/data1/tiansida/data/ReID_origin/mingzhang/'
  datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list', 'data_each_part4.list', 'data_each_part5.list', 'data_each_part6.list', 'data_each_part7.list'] #43928 calss number.
  for di in xrange(len(datafn_list)):
    datafn_list[di] = datapath + datafn_list[di]
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.500.list'
#  data_train = CarReID_Proxy2_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, bucket_key)
  data_train = CarReID_Proxy_Mxnet_Iter2(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn_list, bucket_key)
  
  dlr = 200000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]

  lr_start = 10**-1
  lr_min = 10**-6
  lr_reduce = 0.9
  lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
  lr_stepnum = np.int(np.ceil(lr_stepnum))
  dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
  print 'lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(lr_start, lr_min, lr_reduce, lr_stepnum)
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'

  reid_net = proxy_nca_model.CreateModel_Color2(None, bsz_per_device, proxy_num, data_shape[2:])
#  reid_net_p = proxy_nca_model.CreateModel_Color2_P(None, bsz_per_device, proxy_num, data_shape[2:])


  reid_model = mx.mod.Module(context=ctxs, symbol=reid_net, 
                             label_names=['proxy_yM', 'proxy_ZM'])
#  reid_model_P = mx.mod.Module(context=mx.gpu(1), symbol=reid_net_p, 
#                             label_names=['proxy_yM', 'proxy_ZM'])
#
#  reid_model_P.bind(data_shapes=data_train.provide_data, 
#                    label_shapes=data_train.provide_label,
#                    for_training=False)


  optimizer_params={'learning_rate':0.1,
                    'momentum':0.9,
                    'wd':0.0005,
                    'lr_scheduler':lr_scheduler,
                    'clip_gradient':None,
                    'rescale_grad': 1.0/batch_size}

  proxy_metric = Proxy_Metric()

  if True:
    fn = param_prefix + '_0_' + '.bin'
    reid_model.bind(data_shapes=data_train.provide_data, 
                    label_shapes=data_train.provide_label)
    reid_model.load_params(fn)
    print 'loaded parameters from', fn


  def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)

  mon = mx.mon.Monitor(1, norm_stat, 
                       pattern='.*part1_fc1.*|.*proxy_Z_weight.*')

  def batch_end_call(*args, **kwargs):
  #  print eval_metric.loss_list
    epoch = args[0].epoch
    nbatch = args[0].nbatch + 1
    eval_metric = args[0].eval_metric
    data_batch = args[0].locals['data_batch']  
    if nbatch%show_period==0:
       fn = param_prefix + '_' + str(epoch%4) + '_' + '.bin'
       reid_model.save_params(fn)
       print 'saved parameters into', fn
       if False:
         args, auxs = reid_model.get_params()
         proxy_Z = args['proxy_Z_weight'].asnumpy()
         print 'z:', np.abs(proxy_Z).mean()
         reid_model_P.set_params(args, auxs)
         reid_model_P.forward(data_batch)
         outs = reid_model_P.get_outputs()
         val = outs[1].asnumpy()
         val2 = outs[2].asnumpy()
         val3 = outs[3].asnumpy()
         val4 = outs[4].asnumpy()
         print 'fc1:', np.abs(val).mean(), 'z:', val2.mean(), 'y:', val3, 'one_proxy:', np.abs(val4).mean(), val2[val2>val1]
#       args, auxs = reid_model.get_params()
#       print np.mean(args['proxy_Z_weight'].asnumpy())
#      eval_metric.reset()

  batch_end_calls = [batch_end_call, mx.callback.Speedometer(batch_size, show_period/10)]
  reid_model.fit(train_data=data_train, eval_metric=proxy_metric,
                 optimizer='sgd',
                 optimizer_params=optimizer_params, 
                 initializer=mx.init.Normal(),
                 begin_epoch=0, num_epoch=num_epoch, 
                 eval_end_callback=None,
                 kvstore=None,# monitor=mon,
                 batch_end_callback=batch_end_calls) 


  return 




if __name__=='__main__':
#  Do_Train()
#  Do_Proxy_NCA_Train()
  Do_Proxy_NCA_Train2()
#  Do_Proxy_NCA_Train3()


