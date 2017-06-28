import os
import socket
import sys
import logging
import struct
import fcntl

#def show_local_info():
#  localip = socket.gethostbyname(socket.gethostname())
#  localpath = os.getcwd()
#  role = os.getenv('DMLC_ROLE')
#  logging.info("*********[%s]-%s:%s"%(localip, role, localpath))
##  if role=='server':
##     print "Bye Bye", role
##     exit()
#  return localip

def show_local_info():
  ifname = os.getenv('DMLC_INTERFACE')
  logging.info("+++++++interface name:%s"%(ifname))
  if ifname is None:
    localip = socket.gethostbyname(socket.gethostname())
  else:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    localip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('512s',ifname[:15]))[20:24])
  role = os.getenv('DMLC_ROLE')
  localpath = os.getcwd()
  logging.info("*********[%s]-%s:%s"%(localip, role, localpath))
  return localip


curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, "./distribution"))
localip = show_local_info()

import logging
import numpy as np
import mxnet as mx
from mxnet import metric

from DataIter import CarReID_Proxy_Batch_Mxnet_Iter
from DataIter import CarReID_Proxy_Batch_Plate_Mxnet_Iter2
from DataIter import CarReID_Proxy_Distribution_Batch_Plate_Mxnet_Iter2
from Solver import CarReID_Solver, CarReID_Softmax_Solver, CarReID_Proxy_Solver
from MDL_PARAM import model2 as now_model
#from MDL_PARAM import model2_proxy_nca as proxy_nca_model
#from MDL_PARAM import model3_proxy_nca as proxy_nca_model
from MDL_PARAM import model4_proxy_nca as proxy_nca_model


def save_checkpoint(model, prefix, epoch):
    model.symbol.save('%s-symbol.json' % prefix)
    param_name = '%s-%04d.params' % (prefix, epoch)
    model.save_params(param_name)
    logging.info('[%s]Saved checkpoint to \"%s\"', localip, param_name)

def load_checkpoint(model, prefix, epoch, pZshape):
    param_name = '%s-%04d.params' % (prefix, epoch)
    save_dict = mx.nd.load(param_name)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if name=='proxy_Z_weight':
            sp = pZshape
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
    if model is not None:
        model.set_params(arg_params, aux_params, allow_missing=True)
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
    if False and nbatch%16==0:
      arg_params, aux_params = reid_model.get_params()
      reid_model.set_params(arg_params, aux_params)

    if nbatch%show_period==0:
      #save_checkpoint(reid_model, param_prefix, epoch%4)
      pass


def do_epoch_end_call(param_prefix, epoch, reid_model, \
                      arg_params, aux_params, \
                      reid_model_P, data_train, \
                      proxy_num, proxy_batch):
    if epoch is not None:
       save_checkpoint(reid_model, param_prefix, epoch%4)
       if epoch%4==0: 
           reid_model.pull_ori_params()
           mx.nd.save('proxy_Z_saved.params', [reid_model.ori_parames['proxy_Z_weight']])
           logging.info('Saved original complete proxy_Z into proxy_Z_saved.params.')
       pass

    carnum, proxy_ori_index = data_train.do_reset()

    reid_model.ori_indexes['proxy_Z_weight'][:] = proxy_ori_index
    reid_model.pull_params()
    data_train.reset()
    pass

def Do_Proxy_NCA_Train3():
  print 'Partial Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5), mx.gpu(6)]
#  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
#  ctxs = [mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5), mx.gpu(6), mx.gpu(7)]
#  ctxs = [mx.gpu(2), mx.gpu(1), mx.gpu(3)]
#  ctxs = [mx.gpu(0), mx.gpu(1)]
#  ctxs = [mx.gpu(0)]
  
  devicenum = len(ctxs) 

  num_epoch = 1000000
  batch_size = 56*devicenum
  show_period = 1000

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  print 'batch_size per device:', bsz_per_device
  bucket_key = bsz_per_device

  featdim = 128
  total_proxy_num = 261708#604429#323255#261708#220160#142149#196166#406448#548597
  proxy_batch = 40000
  proxy_num = proxy_batch
  clsnum = proxy_num
  data_shape = (batch_size, 3, 200, 200)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_Z_shape = (proxy_num, featdim)
  proxy_ZM_shape = (batch_size, proxy_num)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
#  datapath = '/home/mingzhang/data/ReID_origin/mingzhang3/' #323255
#  datapath = '/home/mingzhang/data/ReID_origin/mingzhang4/' #604429,#323255
  datapath = '/home/mingzhang/data/ReID_origin/mingzhang5/' #604429,#323255

#  datafn_list = ['front_plate_image_list_train.list', 'back_plate_image_list_train.list'] #261708 calss number.
  datafn_list = ['front_plate_image_list_train.list'] #261708 calss number.
#  datafn_list = ['data_each_part6.list', 'data_each_part7.list'] #142149 calss number.

  for di in xrange(len(datafn_list)):
    datafn_list[di] = datapath + datafn_list[di]
  data_train = CarReID_Proxy_Distribution_Batch_Plate_Mxnet_Iter2(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn_list, total_proxy_num, featdim, proxy_batch, 1)
  
  pcnum = 1 
  dlr = (400000 * pcnum)/batch_size
#  dlr_steps = [dlr, dlr*2, dlr*3, dlr*4]

  lr_start = (10**-3)*1
  lr_min = 10**-5
  lr_reduce = 0.95
  lr_stepnum = np.log(lr_min/lr_start)/np.log(lr_reduce)
  lr_stepnum = np.int(np.ceil(lr_stepnum))
  dlr_steps = [dlr*i for i in xrange(1, lr_stepnum+1)]
  logging.info('pc number:%d, lr_start:%.1e, lr_min:%.1e, lr_reduce:%.2f, lr_stepsnum:%d'%(pcnum, lr_start, lr_min, lr_reduce, lr_stepnum))
  #logging.info(dlr_steps)
  lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(dlr_steps, lr_reduce)
#  param_prefix = 'MDL_PARAM/params2_proxy_nca/car_reid'
#  param_prefix = 'MDL_PARAM/params3_proxy_nca/car_reid'
  param_prefix = 'MDL_PARAM/params4_proxy_nca/car_reid'
  load_paramidx = 3

  reid_net = proxy_nca_model.CreateModel_Color2(None, bsz_per_device, proxy_num, data_shape[2:])


  reid_model = mx.mod.Module(context=ctxs, symbol=reid_net, 
                             label_names=['proxy_yM', 'proxy_ZM'],
                             ori_parames={'proxy_Z_weight':data_train.proxy_Z},
                             ori_indexes={'proxy_Z_weight':data_train.proxy_ori_index})
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

  arg_params = None
  aux_params = None
  allow_missing=False 
  if True and load_paramidx is not None:
    arg_params, aux_params = load_checkpoint(None, param_prefix, load_paramidx, proxy_Z_shape)
    allow_missing=True 

  batch_end_calls = [batch_end_call, mx.callback.Speedometer(batch_size, show_period/20)]
  epoch_all_calls = [epoch_end_call]
  reid_model.fit(train_data=data_train, eval_metric=proxy_metric,
                 optimizer='sgd',
                 optimizer_params=optimizer_params, 
                 initializer=mx.init.Normal(),
                 begin_epoch=0, num_epoch=num_epoch, 
                 eval_end_callback=None,
                 kvstore='dist_async',# monitor=mon,
                 batch_end_callback=batch_end_calls,
                 epoch_end_callback=epoch_all_calls,
                 arg_params=arg_params, aux_params=aux_params,
                 allow_missing=allow_missing) 


  return 

from DataIter import CarReID_Predict_Iter

def prepare_proxy_Z(proxyfn='proxy_Z_gen.params'):
  print 'prepare proxy_Z...'
 
  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5), mx.gpu(6)]
#  ctxs = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
  batchsize = 400 * len(ctxs)
  data_shape = (batchsize, 3, 200, 200)
  label_shape = (batchsize, 1)
  proxy_num = 604429#323255
  featdim = 128
  proxy_Z_shape = (proxy_num, featdim)

#  datapath = '/home/mingzhang/data/ReID_origin/mingzhang3/' #323255
#  datapath = '/home/mingzhang/data/ReID_origin/mingzhang4/' #604429
  datapath = '/home/mingzhang/data/ReID_origin/mingzhang5/' #604429
#  datafn_list = ['front_plate_image_list_train.list', 'back_plate_image_list_train.list'] #261708 calss number.
  datafn_list = ['back_plate_image_list_train.list'] #261708 calss number.
  for di in xrange(len(datafn_list)):
    datafn_list[di] = datapath + datafn_list[di]

  data_predict = CarReID_Predict_Iter(['part1_data'], [data_shape], ['id'], [label_shape], datafn_list) 
  reid_feature_net, _ = proxy_nca_model.CreateModel_Color_Split_test()
  reid_model = mx.mod.Module(context=ctxs, symbol=reid_feature_net, data_names=['part1_data'])
  reid_model.bind(data_shapes=data_predict.provide_data, for_training=False)
#  param_prefix = 'MDL_PARAM/params3_proxy_nca/car_reid'
#  param_prefix = 'MDL_PARAM/params3_proxy_nca.blockmask/car_reid'
  param_prefix = 'MDL_PARAM/params4_proxy_nca.back2/car_reid'
  load_paramidx = 3
  arg_params, aux_params = load_checkpoint(reid_model, param_prefix, load_paramidx, proxy_Z_shape)
  
  proxy_Z = np.random.rand(proxy_num, featdim)-0.5 
  proxy_Z = proxy_Z.astype(np.float32)
  proxy_Z_num = np.zeros(proxy_num, dtype=np.int32)
  for data in data_predict:
    reid_model.forward(data)
    batch_Z = reid_model.get_outputs()[0].asnumpy()
    label = data.label[0].asnumpy().flatten()
    proxy_Z[label, :] = batch_Z
    proxy_Z_num[label] = 1
    num = proxy_Z_num.sum()
    print "batch:%d/%d, proxy_num:%.2f%%(%d/%d)..."%(data_predict.cur_batch, data_predict.num_batches, num*100.0/proxy_num, num, proxy_num)
    if data_predict.cur_batch%100==0:
      savename = proxyfn
      mx.nd.save(savename, [mx.nd.array(proxy_Z)]) 
      print 'saved proxy_Z into file', savename
    if num >= proxy_num:
      print 'initialize proxy_Z is finished...'
      break
  savename = proxyfn
  mx.nd.save(savename, [mx.nd.array(proxy_Z)]) 
  print 'saved proxy_Z into file', savename
  pass


if __name__=='__main__':
#  prepare_proxy_Z("proxy_Z_gen5.params")
#  Do_Train()
  Do_Proxy_NCA_Train3()


