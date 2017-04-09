import logging
import mxnet as mx
import numpy as np
import cPickle
import os
import time


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
#      datainfo1 = dquery['sons'][0]
      id1 = dquery['id']
      for datainfo1 in dquery['sons']:
        data1 = datainfo1['data'].reshape((1,)+datainfo1['data'].shape)
        cmpfile = open('Result/cmp=%s=%s.list'%(id1, datainfo1['name']), 'w')
  #      d1s = np.mean(data1)
  #      print data1.shape, self.arg_params['part1_data'].asnumpy().shape
        self.arg_params['part1_data'][:] = mx.nd.array(data1, self.ctx)
        data_set.reset()
        for dset in data_set:
          id2 = dset['id']
          for datainfo2 in dset['sons']:
            data2 = datainfo2['data'].reshape((1,)+datainfo2['data'].shape)
    #        d2s = np.mean(data2)
            self.arg_params['part2_data'][:] = mx.nd.array(data2, self.ctx)
    
            self.executor.forward(is_train=False)
            cmp_score = self.executor.outputs[0].asnumpy()[0, 0]
            cmpfile.write('%s,%s,%f\n'%(id2, datainfo2['name'], cmp_score)) 
            cmpfile.flush()
    #        print 'query:%s,%.3f,%d; dset:%s,%.3f,%d; %.3f'%(id1, d1s, data_query.cur_idx, id2, d2s, data_set.cur_idx, cmp_score)
            print 'query:%s,%d; dset:%s,%d; %.3f'%(id1, data_query.cur_idx, id2, data_set.cur_idx, cmp_score)
        cmpfile.close()
#       exit()


class CarReID_Feature_Predictor(object):
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
                self.symbol.infer_shape(part1_data=self.data_shape)

    self.arg_params = {}
    for name, shape in zip(arg_names, arg_shapes):
      self.arg_params[name] = mx.nd.zeros(shape, self.ctx)

    aux_names = self.symbol.list_auxiliary_states()
    self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

  def set_params(self, whichone):
    logging.info('loading checkpoint from %s-->%d...', self.prefix, whichone)
    loadfunc = mx.model.load_checkpoint
    _, update_params, aux_params = loadfunc(self.prefix, whichone)
    for name in self.arg_params:
      if name.endswith('weight') or name.endswith('bias') or name.endswith('gamma') or name.endswith('beta'):
        self.arg_params[name][:] = update_params[name]
#      print update_params[name].asnumpy()
    for name in self.aux_params:
      if name.endswith('moving_var') or name.endswith('moving_mean'):
        self.aux_params[name][:] = aux_params[name]   
#        print name, aux_params[name].asnumpy()
#    exit()
    return

  def predict(self, data_set, savepath, whichone=None, logger=None):
    if logger is not None:
      logger.info('Start testing with %s', str(self.ctx))

    self.get_params()
    if whichone is not None:
      self.set_params(whichone)
    self.executor = self.symbol.bind(ctx=self.ctx, args=self.arg_params, grad_req='null', aux_states=self.aux_params)

    # begin training
    data_set.reset()
    for dquery in data_set:
      id1 = dquery['id']
      for datainfo1 in dquery['sons']:
        data1 = datainfo1['data'].reshape((1,)+datainfo1['data'].shape)
        self.arg_params['part1_data'][:] = mx.nd.array(data1, self.ctx)
        self.executor.forward(is_train=False)
        feature = self.executor.outputs[0].asnumpy()
        print feature[0, 0]
        idfolder = savepath + '/' + id1
        if not os.path.exists(idfolder):
          os.makedirs(idfolder)
        featfn = idfolder + '/' + datainfo1['name'] + '.bin'
        cPickle.dump(feature, open(featfn, 'wb')) 
        print 'saved feature:%d/%d, %s'%(data_set.cur_idx, data_set.datalen, featfn)


class CarReID_Compare_Predictor(object):
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
                self.symbol.infer_shape(feature1_data=self.data_shape,
                                        feature2_data=self.data_shape)

    self.arg_params = {}
    for name, shape in zip(arg_names, arg_shapes):
      self.arg_params[name] = mx.nd.zeros(shape, self.ctx)

    aux_names = self.symbol.list_auxiliary_states()
    self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

  def set_params(self, whichone):
    logging.info('loading checkpoint from %s-->%d...', self.prefix, whichone)
    loadfunc = mx.model.load_checkpoint
    _, update_params, aux_params = loadfunc(self.prefix, whichone)
    for name in self.arg_params:
      if name.endswith('weight') or name.endswith('bias') or name.endswith('gamma') or name.endswith('beta'):
        self.arg_params[name][:] = update_params[name]
#      print update_params[name].asnumpy()
    for name in self.aux_params:
      if name.endswith('moving_var') or name.endswith('moving_mean'):
        self.aux_params[name][:] = aux_params[name]   
#        print name, aux_params[name].asnumpy()
#    exit()
    return

  def predict(self, data_query, data_set, whichone=None, logger=None):
    if logger is not None:
      logger.info('Start Comparing with %s', str(self.ctx))

    self.get_params()
    if whichone is not None:
      self.set_params(whichone)
    self.executor = self.symbol.bind(ctx=self.ctx, args=self.arg_params, grad_req='null', aux_states=self.aux_params)

    data_query.reset()
    for dquery in data_query:
      id1 = dquery['ids'][0]
      data1 = dquery['data']
      cmpfile = open('Result/cmp=%s=%s.list'%(id1, dquery['names'][0]), 'w')
      self.arg_params['feature1_data'][:] = mx.nd.array(data1, self.ctx)
      data_set.reset()
      t0 = time.time()
      for dset in data_set:
        id2s = dset['ids']
        data2 = dset['data']
        self.arg_params['feature2_data'][:] = mx.nd.array(data2, self.ctx)
        self.executor.forward(is_train=False)
        cmp_scores = self.executor.outputs[0].asnumpy()
#        print data_set.batchsize, data1.shape, data2.shape, np.sum(np.abs(data1 - data2)), np.sum(cmp_scores)
#        print cmp_scores
#        print data1[0, 0], data2[500, 0]
        if True:
          cmp_scores = np.sum(cmp_scores, axis=1)
#          print data2.shape, cmp_scores
        writestrs = ''
        for bi in xrange(data_set.batchsize):
          id2 = id2s[bi]
          onename = dset['names'][bi]
          cmp_score = cmp_scores[bi]
          writestrs += '%s,%s,%f\n'%(id2, onename, cmp_score)
#          print 'query:%s,%d; dset:%s,%d; %.3f'%(id1, data_query.cur_idx, id2, data_set.cur_idx*data_set.batchsize+bi, cmp_score)
        cmpfile.write(writestrs) 
        cmpfile.flush()
      cmpfile.close()
      t1 = time.time()
      print '%s, %d->time cost:%.3f s'%(id1, data_query.cur_idx, (t1-t0))


class CarReID_Softmax_Predictor(object):
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

  def init_args(self, args):
    for key in args:
      arr = args[key]
      if key.endswith('_weight'):
        self.initializer(key, arr)
      if key.endswith('_bias'):
        arr[:] = 0.0
      if key.endswith('_gamma'):
        arr[:] = 1.0
      if key.endswith('_beta'):
        arr[:] = 0.0
      if key.endswith('_init_c'):
        arr[:] = 0.0
      if key.endswith('_init_h'):
        arr[:] = 0.0

  def get_params(self):
    arg_names = self.symbol.list_arguments()
    arg_shapes, _, aux_shapes = \
                self.symbol.infer_shape(data=self.data_shape)

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


  def predict(self, train_data, showperiod=100, whichone=None, logger=None):
    if logger is not None:
      logger.info('Start softmax predicting with %s', str(self.ctx))

    savefunc = mx.model.save_checkpoint

    self.get_params()
    if whichone is not None:
      self.set_params(whichone)
    self.executor = self.symbol.bind(self.ctx, self.arg_params, grad_req='null', aux_states=self.aux_params)
#    epoch_end_callback = mx.callback.do_checkpoint(self.prefix)
    # begin training
    accus = []
    train_data.reset()
    num_batches = train_data.num_batches
    num_update = 0
    for databatch in train_data:
      num_update += 1
      for k, v in databatch.data.items():
        self.arg_params[k][:] = mx.nd.array(v, self.ctx)
      for k, v in databatch.label.items():
        self.arg_params[k][:] = mx.nd.array(v, self.ctx)
      output_dict = {name: nd for name, nd in zip(self.symbol.list_outputs(), self.executor.outputs)}
      self.executor.forward(is_train=False)

      outval = output_dict['cls_output'].asnumpy()
      label = databatch.label['label']
      cls_predict = np.argmax(outval, axis=1)
      accone = np.mean(cls_predict!=label)
      accus.append(accone)

      if num_update % showperiod == 0:
        print num_update, 'errate_accu:', np.mean(accus)
        accus = []


class CarReID_Compare_Predictor__(object):
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
                self.symbol.infer_shape(feature1_data=self.data_shape,
                                        feature2_data=self.data_shape)

    self.arg_params = {}
    for name, shape in zip(arg_names, arg_shapes):
      self.arg_params[name] = mx.nd.zeros(shape, self.ctx)

    aux_names = self.symbol.list_auxiliary_states()
    self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

  def set_params(self, whichone):
    logging.info('loading checkpoint from %s-->%d...', self.prefix, whichone)
    loadfunc = mx.model.load_checkpoint
    _, update_params, aux_params = loadfunc(self.prefix, whichone)
    for name in self.arg_params:
      if name.endswith('weight') or name.endswith('bias') or name.endswith('gamma') or name.endswith('beta'):
        self.arg_params[name][:] = update_params[name]
#      print update_params[name].asnumpy()
    for name in self.aux_params:
      if name.endswith('moving_var') or name.endswith('moving_mean'):
        self.aux_params[name][:] = aux_params[name]   
#        print name, aux_params[name].asnumpy()
#    exit()
    return

  def predict(self, data_query, data_set, whichone=None, logger=None):
    if logger is not None:
      logger.info('Start Comparing with %s', str(self.ctx))

    self.get_params()
    if whichone is not None:
      self.set_params(whichone)
    self.executor = self.symbol.bind(ctx=self.ctx, args=self.arg_params, grad_req='null', aux_states=self.aux_params)

    data_query.reset()
    for dquery in data_query:
      id1 = dquery['id']
      for datainfo1 in dquery['sons']:
        data1 = datainfo1['data'].reshape((1,)+datainfo1['data'].shape)
        cmpfile = open('Result/cmp=%s=%s.list'%(id1, datainfo1['name']), 'w')
        self.arg_params['feature1_data'][:] = mx.nd.array(data1, self.ctx)
        data_set.reset()
        for dset in data_set:
          id2 = dset['id']
          for datainfo2 in dset['sons']:
            data2 = datainfo2['data'].reshape((1,)+datainfo2['data'].shape)
            self.arg_params['feature2_data'][:] = mx.nd.array(data2, self.ctx)
    
            self.executor.forward(is_train=False)
            cmp_score = self.executor.outputs[0].asnumpy()[0, 0]
            cmpfile.write('%s,%s,%f\n'%(id2, datainfo2['name'], cmp_score)) 
            cmpfile.flush()
            print 'query:%s,%d; dset:%s,%d; %.3f'%(id1, data_query.cur_idx, id2, data_set.cur_idx, cmp_score)
        cmpfile.close()

