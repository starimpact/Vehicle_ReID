import logging
import numpy as np
import mxnet as mx
from mxnet import metric

from DataIter import CarReID_Iter, CarReID_Softmax_Iter, CarReID_Proxy2_Iter
from Solver import CarReID_Solver, CarReID_Softmax_Solver, CarReID_Proxy_Solver
from MDL_PARAM import model2 as now_model
from MDL_PARAM import model2_proxy_nca as proxy_nca_model


def Do_Proxy_NCA_Train():
  print 'Proxy NCA Training...'

  # set up logger
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  
  ctxs = [mx.gpu(0), mx.gpu(1)]
  
  devicenum = len(ctxs) 

  num_epoch = 10000
  batch_size = 64
  featdim = 128
  proxy_num = 43928
  clsnum = proxy_num
  data_shape = (batch_size, 3, 299, 299)
  proxy_yM_shape = (batch_size, proxy_num)
  proxy_Z_shape = (proxy_num, featdim)
  proxy_ZM_shape = (batch_size, proxy_num)
  label_shape = dict(zip(['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape]))
  proxyfn = 'proxy.bin'
  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.list' #43928 calss number.
#  datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data_each.500.list'
  data_train = CarReID_Proxy2_Iter(['data'], [data_shape], ['proxy_yM', 'proxy_ZM'], [proxy_yM_shape, proxy_ZM_shape], datafn)
  
  dlr = 100000/batch_size
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

  assert(batch_size%devicenum==0)
  bsz_per_device = batch_size / devicenum
  print 'batch_size per device:', bsz_per_device
  reid_net = proxy_nca_model.CreateModel_Color2(None, bsz_per_device, proxy_num, data_shape[2:])


  reid_model = mx.model.FeedForward(ctx=ctxs, symbol=reid_net, 
                       begin_epoch=0, num_epoch=num_epoch, epoch_size=None, 
                       learning_rate=0.1, momentum=0.9, wd=0.00005)

  def batch_end_call(*args, **kwargs):
  #  print eval_metric.loss_list
    epoch = args[0].epoch
    nbatch = args[0].nbatch
    eval_metric = args[0].eval_metric
    if nbatch%100==0:
      avgloss = np.mean(eval_metric.loss_list)
      print 'epoch:%d, nbatch:%d, loss:%f'%(epoch, nbatch, avgloss)
      reid_model.save(param_prefix, epoch%4)
      eval_metric.loss_list = []

  proxy_metric = Proxy_Metric()
  reid_model.fit(X=data_train, eval_metric=proxy_metric,
                 eval_end_callback=None,
                 batch_end_callback=batch_end_call) 


  return 



class Proxy_Metric(metric.EvalMetric):
  def __init__(self):
    super(Proxy_Metric, self).__init__('proxy_metric')
    print "hello metric init..."
    self.num = 0
    self.loss_list = []

  def reset(self):
    pass

  def update(self, labels, preds):
    self.num += 1
    loss = preds[0].asnumpy().mean()
    self.loss_list.append(loss)
    

if __name__=='__main__':
#  Do_Train()
  Do_Proxy_NCA_Train()


