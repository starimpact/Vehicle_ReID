import logging

import mxnet as mx
import numpy as np
from collections import namedtuple
import time


# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'eval_metric',
                            'locals'])


def _as_list(obj):
    """A utility function that treat the argument as a list.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list, return it. Otherwise, return `[obj]` as a single-element list.
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


class Module_Info(object):
  def __init__(self, name, symbol, 
               data_names=('data',), 
               data_shapes=(),
               label_names=('label'), 
               label_shapes=(),
               inputs_need_grad=False,
               optimizer='sgd',
               optimizer_params={'learning_rate':0.1, 'momentum':0.9, 'wd':0.0005},
               initializer=mx.init.Normal(),
               context=mx.cpu()):
    self.name = name
    self.symbol = symbol
    self.data_names = data_names
    self.label_names = label_names
    self.data_shapes = data_shapes
    self.label_shapes = label_shapes
    self.inputs_need_grad = inputs_need_grad
    self.optimizer = optimizer
    self.optimizer_params = optimizer_params
    self.initializer = initializer
    self.context = context


class Module_Combine(object):
  def __init__(self, module_infos=[],
               logger=logging):
    assert(len(module_infos)==2, 'For now, only and must support module number 2...')
    self.module_infos = module_infos
    self.logger = logger
    self.modules = []
    for mod_inf in self.module_infos:
      mod = mx.mod.Module(symbol=mod_inf.symbol, 
                          data_names=mod_inf.data_names, 
                          label_names=mod_inf.label_names, 
                          context=mod_inf.context)
      self.modules.append(mod)
    pass

  def bind(self, for_training=False, grad_req='write'):
    for mod_inf, mod in zip(self.module_infos, self.modules):
      datainfo = zip(mod_inf.data_names, mod_inf.data_shapes)
      if mod_inf.label_names is None:
        labelinfo = None
      else:
        labelinfo = zip(mod_inf.label_names, mod_inf.label_shapes)
      mod.bind(data_shapes=datainfo, 
               label_shapes=labelinfo, 
               inputs_need_grad=mod_inf.inputs_need_grad,
               for_training=for_training,
               grad_req=grad_req) 
    pass

  def init_params(self, allow_missing=False, force_init=False):
    for mod_inf, mod in zip(self.module_infos, self.modules):
      mod.init_params(initializer=mod_inf.initializer, 
                      allow_missing=allow_missing,
                      force_init=force_init)

  def set_params(self, arg_aux_list, 
                 allow_missing=False, force_init=True):
    for arg_aux, mod in zip(arg_aux_list, self.modules):
      arg, aux = arg_aux
      mod.set_params(arg, aux, allow_missing, force_init)
    pass

  def get_params(self):
    arg_aux_list = []
    for mod in self.modules:
      arg, aux = mod.get_params()
      arg_aux_list.append((arg, aux))

    return arg_aux_list

  def save_checkpoint(self, prefix, epoch):
    for mod_inf, mod in zip(self.module_infos, self.modules):
      savename = '%s-%s-%04d.params'%(prefix, mod_inf.name, epoch)
      mod.save_params(savename)
      self.logger.info('Saved checkpoint to \"%s\"', savename)
    pass

  def load_checkpoint(self, prefix, epoch):
    for mod_inf, mod in zip(self.module_infos, self.modules):
      savename = '%s-%s-%04d.params'%(prefix, mod_inf.name, epoch)
      mod.load_params(savename)
      self.logger.info('Loaded checkpoint from \"%s\"', savename)
    pass

  def init_optimizer(self, kvstore=None, force_init=False):
    for mod_inf, mod in zip(self.module_infos, self.modules):
      mod.init_optimizer(optimizer=mod_inf.optimizer,
                         optimizer_params=mod_inf.optimizer_params,
                         kvstore=kvstore, force_init=force_init)

  def forward(self, data_batch, is_train=False):
    mod_inf_mods = zip(self.module_infos, self.modules)
    mod_inf0, mod0 = mod_inf_mods[0]
    mod_inf1, mod1 = mod_inf_mods[1]

    data = data_batch.data 
    label = None 
    provide_data = zip(mod_inf0.data_names, mod_inf0.data_shapes)
    if mod_inf0.label_names is None:
      provide_label = None
    else:
      provide_label = zip(mod_inf0.label_names, mod_inf0.label_shapes)
    now_batch = mx.io.DataBatch(data=data, label=label, 
                                provide_data=provide_data,
                                provide_label=provide_label)
    mod0.forward(now_batch, is_train=is_train)

    data = mod0.get_outputs(merge_multi_context=True)
    label = data_batch.label 
    provide_data = zip(mod_inf1.data_names, mod_inf1.data_shapes)
    if mod_inf1.label_names is None:
      provide_label = None 
    else:
      provide_label = zip(mod_inf1.label_names, mod_inf1.label_shapes)
    now_batch = mx.io.DataBatch(data=data, label=label, 
                                provide_data=provide_data,
                                provide_label=provide_label)
    mod1.forward(now_batch, is_train=is_train) 

  def backward(self):
    mod_inf_mods = zip(self.module_infos, self.modules)
    mod_inf0, mod0 = mod_inf_mods[0]
    mod_inf1, mod1 = mod_inf_mods[1]

    mod1.backward()
    pre_input_grads = mod1.get_input_grads(merge_multi_context=True)
    mod0.backward(out_grads=pre_input_grads) 
 

  def forward_backward(self, data_batch):
    self.forward(data_batch, is_train=True)
    self.backward()

  def update(self):
    for mod_inf, mod in zip(self.module_infos, self.modules):
      mod.update() 
    pass

  def update_metric(self, eval_metric, labels):
    modlast = self.modules[-1]
    modlast.update_metric(eval_metric, labels)

  def fit(self, train_data, eval_data=None, eval_metric='acc',
          epoch_end_callback=None, batch_end_callback=None, kvstore=None,
          eval_end_callback=None,
          eval_batch_end_callback=None,
          allow_missing=False,
          force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
          validation_metric=None, monitor=None):
    
    assert num_epoch is not None, 'please specify number of epochs'

    if hasattr(train_data, 'layout_mapper'):
        self.layout_mapper = train_data.layout_mapper
 
    self.bind(for_training=True)

    if monitor is not None:
        self.install_monitor(monitor)
    self.init_params(allow_missing=allow_missing, force_init=force_init)
    self.init_optimizer(kvstore=kvstore)
 
    if validation_metric is None:
        validation_metric = eval_metric
    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)
 
    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        eval_metric.reset()
        for nbatch, data_batch in enumerate(train_data):
            if monitor is not None:
                monitor.tic()
            self.forward_backward(data_batch)
            self.update()
            self.update_metric(eval_metric, data_batch.label)
 
            if monitor is not None:
                monitor.toc_print()
 
            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
 
        # one epoch of training is finished
        for name, val in eval_metric.get_name_value():
            self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
        toc = time.time()
        self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
 
        # sync aux params across devices
        arg_aux_list = self.get_params()
        self.set_params(arg_aux_list)
 
        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, self.symbol, arg_params, aux_params)
 
        #----------------------------------------
        # evaluation on validation set
#        if eval_data:
#            res = self.score(eval_data, validation_metric,
#                             score_end_callback=eval_end_callback,
#                             batch_end_callback=eval_batch_end_callback, epoch=epoch)
#            #TODO: pull this into default
#            for name, val in res:
#                self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
 
        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()













