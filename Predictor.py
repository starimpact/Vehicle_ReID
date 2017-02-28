import logging
import mxnet as mx
import numpy as np


class CarReID_Predictor(object):
  def __init__(self, prefix='', symbol=None, ctx=None, data_shape=None):
    self.prefix = prefix
    self.symbol = symbol
    self.ctx = ctx
    if self.ctx is None:
      self.ctx = mx.cpu() 
    self.data_shape = data_shape
    self.batchsize = data_shape[0]
    self.arg_params = None
    self.aux_params = None
    self.executor = None

  def get_params(self):
    arg_names = self.symbol.list_arguments()
    arg_shapes, _, aux_shapes = \
                self.symbol.infer_shape(part1_data=self.data_shape,
                                        part2_data=self.data_shape)

    self.arg_params = {}
    for name, shape in zip(arg_names, arg_shapes):
      self.arg_params[name] = mx.nd.zeros(shape, self.ctx)

    aux_names = self.symbol.list_auxiliary_states()
    self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

  def set_params(self, whichone):
    logging.info('loading checkpoint from %s-->%d...', self.prefix, whichone)
    loadfunc = mx.model.load_checkpoint
    _, update_params, aux_params = loadfunc(self.prefix, whichone)
    for name in update_params:
      self.arg_params[name][:] = update_params[name]
#      print update_params[name].asnumpy()
    for name in aux_params:
      self.aux_params[name][:] = aux_params[name]   
#      print name, aux_params[name].asnumpy()
#    exit()
    return

  def predict(self, data_query, data_set, whichone=None, logger=None):
    if logger is not None:
      logger.info('Start testing with %s', str(self.ctx))

    self.get_params()
    if whichone is not None:
      self.set_params(whichone)
    self.executor = self.symbol.bind(ctx=self.ctx, args=self.arg_params, grad_req='null', aux_states=self.aux_params)
#    print self.executor.arg_dict['part1_data'], self.executor.arg_dict['part2_data']
#    print self.arg_params['part1_data'], self.arg_params['part2_data']
#    for av in self.aux_params:
#      print av, self.aux_params[av].asnumpy()
#    exit()
    # begin training
    data_query.reset()
    for dquery in data_query:
      data1 = dquery['sons'][0]
      id1 = dquery['id']
      data1 = data1.reshape((1,)+data1.shape)
      cmpfile = open('Result/cmp_%s_0.list'%(id1), 'w')
#      d1s = np.mean(data1)
#      print data1.shape, self.arg_params['part1_data'].asnumpy().shape
      self.arg_params['part1_data'][:] = mx.nd.array(data1, self.ctx)
      data_set.reset()
      for dset in data_set:
        id2 = dset['id']
        for si, data2 in enumerate(dset['sons']):
          data2 = data2.reshape((1,)+data2.shape)
  #        d2s = np.mean(data2)
          self.arg_params['part2_data'][:] = mx.nd.array(data2, self.ctx)
  
          self.executor.forward(is_train=False)
          cmp_score = self.executor.outputs[0].asnumpy()[0, 0]
          cmpfile.write('%s,%d,%f\n'%(id2, si, cmp_score)) 
          cmpfile.flush()
  #        print 'query:%s,%.3f,%d; dset:%s,%.3f,%d; %.3f'%(id1, d1s, data_query.cur_idx, id2, d2s, data_set.cur_idx, cmp_score)
          print 'query:%s,%d; dset:%s,%d; %.3f'%(id1, data_query.cur_idx, id2, data_set.cur_idx, cmp_score)
      cmpfile.close()
#       exit()




