import logging
import numpy as np
import mxnet as mx
from mxnet import metric

from DataIter import CarReID_Proxy_Mxnet_Iter
from DataIter import CarReID_Proxy_Mxnet_Iter2
from Module_Combine import Module_Info, Module_Combine
from MDL_PARAM import model2_proxy_nca_combine as proxy_nca_combine



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
  
  mod_context0 = [mx.gpu(1), mx.gpu(2), mx.gpu(3)]
  mod_context1 = [mx.gpu(0)]
  
  devicenum = len(mod_context0) 
  proxy_devicenum = len(mod_context1) 

  num_epoch = 10000
  batch_size = 50*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  proxy_bsz_per_device = batch_size / proxy_devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  proxy_num = 43928
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_ZM_shape = (batch_size, proxy_num)
  reid_feature_shape = (batch_size, featdim)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.list' #43928 calss number.
  data_train = CarReID_Proxy_Mxnet_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, bucket_key)
  
  dlr = 400000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]

  lr_start = 0.6*(10**-4)
  lr_min = 10**-5
  lr_reduce = 0.95
  lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
  lr_stepnum = np.int(np.ceil(lr_stepnum))
  dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
  print 'lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(lr_start, lr_min, lr_reduce, lr_stepnum)
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params2_proxy_nca_combine/car_reid'

  reid_feature, proxy_loss = proxy_nca_combine.CreateModel_Color_Combine(None, bsz_per_device, proxy_bsz_per_device, proxy_num, data_shape[2:])

  optimizer_params={'learning_rate':lr_start,
                    'momentum':0.9,
                    'wd':0.0005,
                    'lr_scheduler':lr_scheduler,
                    'clip_gradient':None,
                    'rescale_grad': 1.0/batch_size}

  mod_info0 = Module_Info(name='reid_feature', symbol=reid_feature,
                          data_names=['data'], data_shapes=[data_shape],
                          label_names=None, label_shapes=None,
                          inputs_need_grad=False,
                          optimizer='sgd',
                          optimizer_params=optimizer_params,
                          context=mod_context0) 

  mod_info1 = Module_Info(name='proxy_loss', symbol=proxy_loss,
                          data_names=['reid_feature'], data_shapes=[reid_feature_shape],
                          label_names=['proxy_yM', 'proxy_ZM'], label_shapes=[proxy_yM_shape, proxy_ZM_shape],
                          inputs_need_grad=True,
                          optimizer='sgd',
                          optimizer_params=optimizer_params,
                          context=mod_context1)

  reid_model = Module_Combine(module_infos=[mod_info0, mod_info1])

  proxy_metric = Proxy_Metric()

  if True:
    reid_model.bind(for_training=True)
    reid_model.load_checkpoint(param_prefix, 1)


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
       reid_model.save_checkpoint(param_prefix, epoch%4)

  batch_end_calls = [batch_end_call, mx.callback.Speedometer(batch_size, show_period/10)]
  reid_model.fit(train_data=data_train, eval_metric=proxy_metric,
                 begin_epoch=50, num_epoch=num_epoch, 
                 eval_end_callback=None,
                 batch_end_callback=batch_end_calls) 


  return 


def Do_Proxy_NCA_Train3():
  print 'Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  mod_context0 = [mx.gpu(7), mx.gpu(6), mx.gpu(5)]
  mod_context1 = [mx.gpu(4), mx.gpu(3), mx.gpu(2), mx.gpu(1), mx.gpu(0)]

  devicenum = len(mod_context0) 
  proxy_devicenum = len(mod_context1) 

  num_epoch = 10000
  batch_size = 50*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  proxy_bsz_per_device = batch_size / proxy_devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  proxy_num = 196166#294255#548597
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_ZM_shape = (batch_size, proxy_num)
  reid_feature_shape = (batch_size, featdim)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
  datapath = '/home/mingzhang/data/ReID_origin/mingzhang/'
#  datapath = '/mnt/sdc1/mingzhang/ReID_origin/mingzhang/'
 # datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list', 'data_each_part4.list', 'data_each_part5.list', 'data_each_part6.list', 'data_each_part7.list'] #43928 calss number.
 # datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list', 'data_each_part4.list'] #43928 calss number.
  datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list'] #43928 calss number.
  for di in xrange(len(datafn_list)):
    datafn_list[di] = datapath + datafn_list[di]
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.500.list'
#  data_train = CarReID_Proxy2_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, bucket_key)
  data_train = CarReID_Proxy_Mxnet_Iter2(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn_list, bucket_key)
  
  dlr = 400000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]

  lr_start = (10**-3)
  lr_min = 10**-5
  lr_reduce = 0.95
  lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
  lr_stepnum = np.int(np.ceil(lr_stepnum))
  dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
  print 'lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(lr_start, lr_min, lr_reduce, lr_stepnum)
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)
#  lr_scheduler = mx.lr_scheduler.FactorScheduler(dlr, 0.9)
  param_prefix = 'MDL_PARAM/params2_proxy_nca_combine/car_reid'

  reid_feature, proxy_loss = proxy_nca_combine.CreateModel_Color_Combine(None, bsz_per_device, proxy_bsz_per_device, proxy_num, data_shape[2:])

  optimizer_params={'learning_rate':lr_start,
                    'momentum':0.9,
                    'wd':0.0005,
                    'lr_scheduler':lr_scheduler,
                    'clip_gradient':None,
                    'rescale_grad': 1.0/batch_size}

  mod_info0 = Module_Info(name='reid_feature', symbol=reid_feature,
                          data_names=['data'], data_shapes=[data_shape],
                          label_names=None, label_shapes=None,
                          inputs_need_grad=False,
                          optimizer='sgd',
                          optimizer_params=optimizer_params,
                          context=mod_context0) 

  mod_info1 = Module_Info(name='proxy_loss', symbol=proxy_loss,
                          data_names=['reid_feature'], data_shapes=[reid_feature_shape],
                          label_names=['proxy_yM', 'proxy_ZM'], label_shapes=[proxy_yM_shape, proxy_ZM_shape],
                          inputs_need_grad=True,
                          optimizer='sgd',
                          optimizer_params=optimizer_params,
                          context=mod_context1)

  reid_model = Module_Combine(module_infos=[mod_info0, mod_info1])

  proxy_metric = Proxy_Metric()

  if True:
    reid_model.bind(for_training=True)
    reid_model.load_checkpoint(param_prefix, 1)

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
       reid_model.save_checkpoint(param_prefix, epoch%4)

  batch_end_calls = [batch_end_call, mx.callback.Speedometer(batch_size, show_period/10)]
  reid_model.fit(train_data=data_train, eval_metric=proxy_metric,
                 begin_epoch=10, num_epoch=num_epoch, 
                 eval_end_callback=None, kvstore=None,
                 batch_end_callback=batch_end_calls) 


  return 




if __name__=='__main__':
#  Do_Train()
#  Do_Proxy_NCA_Train()
#  Do_Proxy_NCA_Train2()
  Do_Proxy_NCA_Train3()


