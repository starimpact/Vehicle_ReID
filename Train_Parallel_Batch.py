import logging
import numpy as np
import mxnet as mx
from mxnet import metric

from DataIter import CarReID_Proxy_Batch_Mxnet_Iter
from DataIter import CarReID_Proxy_Batch_Mxnet_Iter2
from Solver import CarReID_Solver, CarReID_Softmax_Solver, CarReID_Proxy_Solver
from MDL_PARAM import model2 as now_model
from MDL_PARAM import model2_proxy_nca as proxy_nca_model

def save_checkpoint(model, prefix, epoch):
    model.symbol.save('%s-symbol.json' % prefix)
    param_name = '%s-%04d.params' % (prefix, epoch)
    model.save_params(param_name)
    logging.info('Saved checkpoint to \"%s\"', param_name)

def load_checkpoint2(model, prefix, epoch):
#    symbol = mx.sym.load('%s-symbol.json' % prefix)
    param_name = '%s-%04d.params' % (prefix, epoch)
    model.load_params(param_name)
    arg_params, aux_params = model.get_params()
    logging.info('Load checkpoint from \"%s\"', param_name)
    return arg_params, aux_params

def load_checkpoint(model, prefix, epoch):
    param_name = '%s-%04d.params' % (prefix, epoch)
    save_dict = mx.nd.load(param_name)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if name=='proxy_Z_weight':
            sp = value.shape
            rndv = np.random.rand(*sp)-0.5
            arg_params[name] = mx.nd.array(rndv)
            print 'skipped %s...'%name
            continue
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


class Proxy_Metric(metric.EvalMetric):
  def __init__(self, saveperiod=1, batch_hardidxes=[]):
    print "hello metric init..."
    super(Proxy_Metric, self).__init__('proxy_metric', 1)
    self.p_inst = 0
    self.saveperiod=saveperiod
    self.batch_hardidxes = batch_hardidxes

  def update(self, labels, preds):
#    print '=========%d========='%(self.p_inst)
    self.p_inst += 1
    for i in xrange(self.num):
      self.num_inst[i] += 1
    eachloss = preds[0].asnumpy()
    loss = eachloss.mean()
    self.sum_metric[0] += loss
#    print loss, len(preds)#, labels
#    self.sum_metric[1] += np.sum(eachloss<=0.0)
#    self.sum_metric[2] += np.sum(eachloss>0.0)
#    if loss < 0: self.batch_hardidxes[:] = eachloss
#    for bi in xrange(len(eachloss)):
#      oneloss = eachloss[bi]
#      if oneloss*5 > loss:#store harder example
#        self.batch_hardidxes.append([bi, oneloss])
    

def do_batch_end_call(reid_model, param_prefix, \
                      show_period, \
                      batch_hardidxes, \
                      *args, **kwargs):
  #  print eval_metric.loss_list
    epoch = args[0].epoch
    nbatch = args[0].nbatch + 1
    eval_metric = args[0].eval_metric
    data_batch = args[0].locals['data_batch']  
    train_data = args[0].locals['train_data']  
    
    #synchronize parameters in small period.
    if nbatch%8==0:
      arg_params, aux_params = reid_model.get_params()
      reid_model.set_params(arg_params, aux_params)

    if nbatch%show_period==0:
       save_checkpoint(reid_model, param_prefix, epoch%4)



def do_epoch_end_call(param_prefix, epoch, reid_model, \
                      arg_params, aux_params, \
                      reid_model_P, data_train, \
                      proxy_num, proxy_batch):
    if epoch is not None:
       save_checkpoint(reid_model, param_prefix, epoch%4)

    proxy_Z_now = arg_params['proxy_Z_weight']
    if epoch is not None:
      data_train.proxy_updateset(proxy_Z_now)
    carnum, proxy_Zfeat = data_train.do_reset()

    proxy_Z_now[:] = proxy_Zfeat
    reid_model.set_params(arg_params, aux_params)
    data_train.reset()
    pass



def Do_Proxy_NCA_Train2():
  print 'Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
#  ctxs = [mx.gpu(2), mx.gpu(1), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1)]
#  ctxs = [mx.gpu(0)]
  
  devicenum = len(ctxs) 

  num_epoch = 1000000
  batch_size = 32*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  total_proxy_num = 43928
  proxy_batch = 20000#720000
  proxy_num = proxy_batch#43928#proxy_batch
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_Z_shape = (proxy_num, featdim)
  proxy_ZM_shape = (batch_size, proxy_num)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.list' #43928 calss number.
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.500.list'
#  data_train = CarReID_Proxy2_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, bucket_key)
  data_train = CarReID_Proxy_Batch_Mxnet_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn, total_proxy_num, featdim, proxy_batch)
  
  dlr = 200000/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]

  lr_start = (10**-1)
  lr_min = 10**-5
  lr_reduce = 0.95
  lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
  lr_stepnum = np.int(np.ceil(lr_stepnum))
  dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
  print 'lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(lr_start, lr_min, lr_reduce, lr_stepnum)
  print dlr_steps
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'
  load_paramidx = None 

  reid_net = proxy_nca_model.CreateModel_Color2(None, bsz_per_device, proxy_num, data_shape[2:])


  reid_model = mx.mod.Module(context=ctxs, symbol=reid_net, 
                             label_names=['proxy_yM', 'proxy_ZM'])
#

  optimizer_params={'learning_rate':lr_start,
                    'momentum':0.9,
                    'wd':0.0005,
                    'lr_scheduler':lr_scheduler,
                    'clip_gradient':None,
                    'rescale_grad': 1.0/batch_size}

  batch_hardidxes = []
  proxy_metric = Proxy_Metric(batch_hardidxes=batch_hardidxes)


  def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)

  mon = mx.mon.Monitor(1, norm_stat, 
                       pattern='.*part1_fc1.*|.*proxy_Z_weight.*')

  def batch_end_call(*args, **kwargs):
    do_batch_end_call(reid_model, param_prefix, \
                      show_period, \
                      batch_hardidxes, \
                      *args, **kwargs)


  def epoch_end_call(epoch, symbol, arg_params, aux_params):
    do_epoch_end_call(param_prefix, epoch, reid_model, \
                      arg_params, aux_params, \
                      None, data_train,\
                      proxy_num, proxy_batch)

  if True and load_paramidx is not None :
    reid_model.bind(data_shapes=data_train.provide_data, 
                    label_shapes=data_train.provide_label)
    arg_params, aux_params = load_checkpoint(reid_model, param_prefix, load_paramidx)
    epoch_end_call(None, None, arg_params, aux_params)

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
  print 'Partial Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  

#  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5), mx.gpu(6), mx.gpu(7)]
  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
#  ctxs = [mx.gpu(2), mx.gpu(1), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1)]
#  ctxs = [mx.gpu(0)]
  
  devicenum = len(ctxs) 

  num_epoch = 1000000
  batch_size = 32*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  total_proxy_num = 142149#196166#406448#548597
  proxy_batch = 40000
  proxy_num = proxy_batch
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_Z_shape = (proxy_num, featdim)
  proxy_ZM_shape = (batch_size, proxy_num)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
#  datapath = '/home/mingzhang/data/ReID_origin/mingzhang/'
#  datapath = '/mnt/ssd2/minzhang/ReID_origin/mingzhang/'
  datapath = '/mnt/ssd2/minzhang/ReID_origin/mingzhang2/'

#  datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list', 'data_each_part4.list', 'data_each_part5.list', 'data_each_part6.list', 'data_each_part7.list'] #43928 calss number.
#  datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list', 'data_each_part4.list', 'data_each_part5.list'] #406448 calss number.
#  datafn_list = ['data_each_part1.list', 'data_each_part2.list', 'data_each_part3.list'] #196166 calss number.
  datafn_list = ['data_each_part6.list', 'data_each_part7.list'] #142149 calss number.
#  datafn_list = ['data_each_part1.list'] #43912 calss number.

  for di in xrange(len(datafn_list)):
    datafn_list[di] = datapath + datafn_list[di]
  data_train = CarReID_Proxy_Batch_Mxnet_Iter2(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn_list, total_proxy_num, featdim, proxy_batch, 1)
  
  dlr = 800000/batch_size
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
  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'
  load_paramidx = 3

  reid_net = proxy_nca_model.CreateModel_Color2(None, bsz_per_device, proxy_num, data_shape[2:])


  reid_model = mx.mod.Module(context=ctxs, symbol=reid_net, 
                             label_names=['proxy_yM', 'proxy_ZM'])
#

  optimizer_params={'learning_rate':lr_start,
                    'momentum':0.9,
                    'wd':0.0005,
                    'lr_scheduler':lr_scheduler,
                    'clip_gradient':None,
                    'rescale_grad': 1.0/batch_size}

  batch_hardidxes = []
  proxy_metric = Proxy_Metric(batch_hardidxes=batch_hardidxes)

  def norm_stat(d):
    return mx.nd.norm(d)/np.sqrt(d.size)

  mon = mx.mon.Monitor(1, norm_stat, 
                       pattern='.*part1_fc1.*|.*proxy_Z_weight.*')

  def batch_end_call(*args, **kwargs):
    do_batch_end_call(reid_model, param_prefix, \
                      show_period, \
                      batch_hardidxes, \
                      *args, **kwargs)

  def epoch_end_call(epoch, symbol, arg_params, aux_params):
    do_epoch_end_call(param_prefix, epoch, reid_model, \
                      arg_params, aux_params, \
                      None, data_train, \
                      proxy_num, proxy_batch) 

  if True and load_paramidx is not None :
    reid_model.bind(data_shapes=data_train.provide_data, 
                    label_shapes=data_train.provide_label)
    arg_params, aux_params = load_checkpoint(reid_model, param_prefix, load_paramidx)
    do_epoch_end_call(param_prefix, None, reid_model, \
                      arg_params, aux_params, \
                      None, data_train, \
                      proxy_num, proxy_batch)
#    epoch_end_call(None, None, arg_params, aux_params)


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




if __name__=='__main__':
#  Do_Train()
#  Do_Proxy_NCA_Train()
#  Do_Proxy_NCA_Train2()
  Do_Proxy_NCA_Train3()


