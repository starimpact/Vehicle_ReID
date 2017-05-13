import numpy as np
import mxnet as mx 
import DataGenerator as dg 
import operator
import os

datafn = '/media/data1/mzhang/data/car_ReID_for_zhangming/data.list'
datafn = '/home/mingzhang/data/car_ReID_for_zhangming/data.list'

class CarReID_Iter(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, 
               data_label_gen=None):
    super(CarReID_Iter, self).__init__()

    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.data_label_gen = data_label_gen
    self.cur_batch = 0
#    self.datas_labels = self.data_label_gen(self._provide_data, self._provide_label) 
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.rndidx_list = np.random.permutation(self.datalen)
    self.num_batches = self.datalen / label_shapes[0][0]

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.rndidx_list = np.random.permutation(self.datalen)

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def next(self):
    if self.cur_batch < self.num_batches:
      self.cur_batch += 1
      datas, labels = dg.get_pairs_data_label(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.cur_batch) 
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration





class CarReID_Test_Iter(mx.io.DataIter):
  def __init__(self, data_name, data_shape, datafn, normalize=True):
    super(CarReID_Test_Iter, self).__init__()

    self._provide_data = zip(data_name, data_shape)
    self.cur_idx = 0
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.normalize = normalize

  def __iter__(self):
    return self

  def reset(self):
    self.cur_idx = 0

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  def next(self):
    if self.cur_idx < self.datalen:
#      print self._provide_data
      carinfo = dg.get_data_label_test(self._provide_data[0][1][1:], self.datalist, self.cur_idx, self.normalize) 
      self.cur_idx += 1
      return carinfo 
    else:
      raise StopIteration


class CarReID_Feat_Query_Iter(mx.io.DataIter):
  def __init__(self, data_name, data_shape, datafn):
    super(CarReID_Feat_Query_Iter, self).__init__()

    self._provide_data = zip(data_name, data_shape)
    self.cur_idx = 0
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.batchsize = data_shape[0][0]
    self.batchnum = self.datalen

  def __iter__(self):
    return self

  def reset(self):
    self.cur_idx = 0

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  def next(self):
    if self.cur_idx < self.batchnum:
#      print self._provide_data
      carinfo = dg.get_feature_label_query_test(self._provide_data[0][1], self.datalist, self.cur_idx) 
      self.cur_idx += 1
      return carinfo 
    else:
      raise StopIteration



class CarReID_Feat_Iter(mx.io.DataIter):
  def __init__(self, data_name, data_shape, datafn):
    super(CarReID_Feat_Iter, self).__init__()

    self._provide_data = zip(data_name, data_shape)
    self.cur_idx = 0
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.batchsize = data_shape[0][0]
    self.batchnum = self.datalen / self.batchsize

  def __iter__(self):
    return self

  def reset(self):
    self.cur_idx = 0

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  def next(self):
    if self.cur_idx < self.batchnum:
#      print self._provide_data
      carinfo = dg.get_feature_label_test(self._provide_data[0][1], self.datalist, self.cur_idx) 
      self.cur_idx += 1
      return carinfo 
    else:
      raise StopIteration


class CarReID_Softmax_Iter(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn):
    super(CarReID_Softmax_Iter, self).__init__()

    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.cur_batch = 0
#    self.datas_labels = self.data_label_gen(self._provide_data, self._provide_label) 
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.rndidx_list = np.random.permutation(self.datalen)
    self.num_batches = self.datalen / label_shapes[0][0]

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.rndidx_list = np.random.permutation(self.datalen)

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def next(self):
    if self.cur_batch < self.num_batches:
      datas, labels = dg.get_data_label(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.cur_batch) 
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration


class CarReID_Proxy_Iter(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn, proxyfn):
    super(CarReID_Proxy_Iter, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.cur_batch = 0
#    self.datas_labels = self.data_label_gen(self._provide_data, self._provide_label) 
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.rndidx_list = np.random.permutation(self.datalen)
    self.num_batches = self.datalen / label_shapes[0][0]
    self.labeldict = dict(self._provide_label)
    self.proxy_set = None#dg.get_proxyset(proxyfn, self.labeldict['proxy_Z'])

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.rndidx_list = np.random.permutation(self.datalen)

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def next(self):
    if self.cur_batch < self.num_batches:
      datas, labels = dg.get_data_label_proxy(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.proxy_set, self.cur_batch) 
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration


class CarReID_Proxy2_Iter(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn, bucket_key):
    super(CarReID_Proxy2_Iter, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.cur_batch = 0
#    self.datas_labels = self.data_label_gen(self._provide_data, self._provide_label) 
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.rndidx_list = np.random.permutation(self.datalen)
    self.num_batches = self.datalen / label_shapes[0][0]
    self.labeldict = dict(self._provide_label)
    self.default_bucket_key = bucket_key

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.rndidx_list = np.random.permutation(self.datalen)

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def next(self):
    if self.cur_batch < self.num_batches:
      datas, labels = dg.get_data_label_proxy2(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.cur_batch) 
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration


class CarReID_Proxy_Mxnet_Iter(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn, bucket_key):
    super(CarReID_Proxy_Mxnet_Iter, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.datas_batch = {} 
    self.datas_batch['data'] = mx.nd.zeros(data_shapes[0], dtype=np.float32)
    self.datas_batch['databuffer'] = np.zeros(data_shapes[0], dtype=np.float32)
    self.labels_batch = {}
    self.labels_batch['proxy_yM'] = mx.nd.zeros(label_shapes[0], dtype=np.float32)
    self.labels_batch['proxy_ZM'] = mx.nd.zeros(label_shapes[1], dtype=np.float32)
    self.cur_batch = 0
#    self.datas_labels = self.data_label_gen(self._provide_data, self._provide_label) 
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.rndidx_list = range(self.datalen)#np.random.permutation(self.datalen)
    self.num_batches = self.datalen / label_shapes[0][0]
    self.labeldict = dict(self._provide_label)
    self.default_bucket_key = bucket_key

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
#    self.rndidx_list = np.random.permutation(self.datalen)

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def next(self):
    if self.cur_batch < 1:#self.num_batches:
#      datas, labels = dg.get_data_label_proxy_mxnet(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.cur_batch) 
      datas, labels = dg.get_data_label_proxy_mxnet_threads(self._provide_data, self.datas_batch, self._provide_label, self.labels_batch, self.datalist, self.rndidx_list, self.cur_batch) 
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration


class CarReID_Proxy_Batch_Mxnet_Iter(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn, 
               proxy_num, featdim, proxy_batchsize, repeat_times=4, num_proxy_batch_max=0.0):
    super(CarReID_Proxy_Batch_Mxnet_Iter, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.datas_batch = {} 
    self.datas_batch['data'] = mx.nd.zeros(data_shapes[0], dtype=np.float32)
    self.datas_batch['databuffer'] = np.zeros(data_shapes[0], dtype=np.float32)
    self.labels_batch = {}
    self.labels_batch['proxy_yM'] = mx.nd.zeros(label_shapes[0], dtype=np.float32)
    self.labels_batch['proxy_ZM'] = mx.nd.zeros(label_shapes[1], dtype=np.float32)
    self.cur_batch = 0
    self.datalist = dg.get_datalist(datafn)
    self.datalen = len(self.datalist)
    self.labeldict = dict(self._provide_label)
    self.proxy_batchsize = proxy_batchsize
    self.rndidx_list = None 
    self.num_batches = self.proxy_batchsize / label_shapes[0][0]
    self.batch_carids = []
    self.batch_infos = []
    self.num_proxy_batch = self.datalen / self.proxy_batchsize
    self.num_proxy_batch_max = num_proxy_batch_max
    self.cur_proxy_batch = 0
    self.big_epoch = 0
    self.proxy_num = proxy_num
    self.featdim = featdim
    self.proxy_Z_fn = './proxy_Z.params'
    proxy_Ztmp = np.random.rand(self.proxy_num, self.featdim)-0.5
    self.proxy_Z = proxy_Ztmp.astype(np.float32) 
    if os.path.exists(self.proxy_Z_fn):
      tmpZ = mx.nd.load(self.proxy_Z_fn)
      self.proxy_Z = tmpZ[0].asnumpy()
      assert(self.proxy_num==tmpZ[0].shape[0])
      print 'load proxy_Z from', self.proxy_Z_fn
    proxy_Z_ptmp = np.random.rand(self.proxy_batchsize, self.featdim)-0.5
    self.proxy_Z_p = proxy_Z_ptmp.astype(np.float32)
    self.proxy_Z_map = np.zeros(self.proxy_batchsize, dtype=np.int32)-1
    self.caridnum = None
    self.total_proxy_batch_epoch = 0
    self.repeat_times = repeat_times
    self.do_reset()

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.batch_carids = []
    self.batch_infos = []
    pass

  def proxy_updateset(self, proxy_Z_p_new):
    num = np.sum(self.proxy_Z_map>-1)
    p_Z = proxy_Z_p_new.asnumpy()
    self.proxy_Z_p[:] = p_Z
    for i in xrange(num):
      carid = self.proxy_Z_map[i]
      self.proxy_Z[carid] = p_Z[i]
    savename = self.proxy_Z_fn 
    mx.nd.save(savename, [mx.nd.array(self.proxy_Z)])
#    a = self.proxy_Z
#    a = p_Z
    print 'save proxy_Z into file', savename#, a#, a[a<0], a[a>0]
    pass

  def do_reset(self):
    self.cur_batch = 0        
    self.batch_carids = []
    self.batch_infos = []
    if self.total_proxy_batch_epoch == 0 \
       or self.cur_proxy_batch == self.num_proxy_batch \
       or (self.num_proxy_batch_max > 0.0 \
       and self.cur_proxy_batch > self.num_proxy_batch * self.num_proxy_batch_max):
      self.cur_proxy_batch = 0 
      self.big_epoch += 1
      self.rndidx_list = np.random.permutation(self.datalen)

    self.proxy_datalist = []
    carids = {}
    self.proxy_Z_map[:] = -1
    prndidxs = np.random.permutation(self.proxy_batchsize)
    for i in xrange(self.proxy_batchsize):
      pidx = prndidxs[i]
      pxyi = self.cur_proxy_batch * self.proxy_batchsize + pidx
      idx = self.rndidx_list[pxyi]
      onedata = self.datalist[idx] 
      parts = onedata.split(',')
      path = parts[0]
      son = parts[1]
      carid = path.split('/')[-1]
      if not carids.has_key(carid):
        carids[carid] = len(carids)
      ori_id = int(carid)
      proxyid = carids[carid] 
      self.proxy_Z_p[proxyid] = self.proxy_Z[ori_id]
      self.proxy_Z_map[proxyid] = ori_id
      proxy_str = '%s,%s,%s,%s'%(path, son, carid, str(proxyid))
      self.proxy_datalist.append(proxy_str)

    self.caridnum = len(carids)
    print 'got another proxy batch to train(%d/%d/%d, %d/%d) [big_epoch=%d]...'%(\
         self.caridnum, self.proxy_batchsize, self.datalen, self.cur_proxy_batch+1,\
         self.num_proxy_batch, self.big_epoch)

    self.total_proxy_batch_epoch += 1
    if self.total_proxy_batch_epoch%self.repeat_times==0: 
      self.cur_proxy_batch += 1
#    print self.proxy_Z_p, self.proxy_Z_p.sum()
    return self.caridnum, self.proxy_Z_p

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label


  def next(self):
    if self.cur_batch < self.num_batches:
#      datas, labels, carids, infos = dg.get_data_label_proxy_batch_mxnet(self._provide_data, self._provide_label, self.proxy_datalist, self.cur_batch) 
      datas, labels, carids, infos = dg.get_data_label_proxy_batch_mxnet_threads(self._provide_data, self.datas_batch, self._provide_label, self.labels_batch, self.proxy_datalist, self.cur_batch, self.caridnum) 
      self.batch_carids = carids
      self.batch_infos = infos
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration




class CarReID_Proxy_Mxnet_Iter2(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn_list, bucket_key):
    super(CarReID_Proxy_Mxnet_Iter2, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.datas_batch = {} 
    self.datas_batch['data'] = mx.nd.zeros(data_shapes[0], dtype=np.float32)
    self.datas_batch['databuffer'] = np.zeros(data_shapes[0], dtype=np.float32)
    self.labels_batch = {}
    self.labels_batch['proxy_yM'] = mx.nd.zeros(label_shapes[0], dtype=np.float32)
    self.labels_batch['proxy_ZM'] = mx.nd.zeros(label_shapes[1], dtype=np.float32)
    self.cur_batch = 0
#    self.datas_labels = self.data_label_gen(self._provide_data, self._provide_label) 
    self.datalist = dg.get_datalist2(datafn_list)
    self.datalen = len(self.datalist)
    self.rndidx_list = np.random.permutation(self.datalen)
    self.num_batches = self.datalen / label_shapes[0][0]
    self.labeldict = dict(self._provide_label)
    self.default_bucket_key = bucket_key

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.rndidx_list = np.random.permutation(self.datalen)

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def next(self):
    if self.cur_batch < self.num_batches:
#      datas, labels = dg.get_data_label_proxy_mxnet2(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.cur_batch) 
      datas, labels = dg.get_data_label_proxy_mxnet2_threads(self._provide_data, self.datas_batch, self._provide_label, self.labels_batch, self.datalist, self.rndidx_list, self.cur_batch) 
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration


class CarReID_Proxy_Batch_Mxnet_Iter2(mx.io.DataIter):
  def __init__(self, data_names, data_shapes, label_names, label_shapes, datafn, 
               proxy_num, featdim, proxy_batchsize, repeat_times=4, num_proxy_batch_max=0.0):
    super(CarReID_Proxy_Batch_Mxnet_Iter2, self).__init__()

    self.batch_size = data_shapes[0][0]
    self._provide_data = zip(data_names, data_shapes)
    self._provide_label = zip(label_names, label_shapes)
    self.datas_batch = {} 
    self.datas_batch['data'] = mx.nd.zeros(data_shapes[0], dtype=np.float32)
    self.datas_batch['databuffer'] = np.zeros(data_shapes[0], dtype=np.float32)
    self.labels_batch = {}
    self.labels_batch['proxy_yM'] = mx.nd.zeros(label_shapes[0], dtype=np.float32)
    self.labels_batch['proxy_ZM'] = mx.nd.zeros(label_shapes[1], dtype=np.float32)
    self.cur_batch = 0
    self.datalist = dg.get_datalist2(datafn)
    self.datalen = len(self.datalist)
    self.labeldict = dict(self._provide_label)
    self.proxy_batchsize = proxy_batchsize
    self.rndidx_list = None 
    self.num_batches = self.proxy_batchsize / label_shapes[0][0]
    self.batch_carids = []
    self.batch_infos = []
    self.num_proxy_batch = self.datalen / self.proxy_batchsize
    self.num_proxy_batch_max = num_proxy_batch_max
    self.cur_proxy_batch = 0
    self.big_epoch = 0
    self.proxy_num = proxy_num
    self.featdim = featdim
    self.proxy_Z_fn = './proxy_Z.params'
    proxy_Ztmp = np.random.rand(self.proxy_num, self.featdim)-0.5
    self.proxy_Z = proxy_Ztmp.astype(np.float32) 
    if os.path.exists(self.proxy_Z_fn):
      tmpZ = mx.nd.load(self.proxy_Z_fn)
      self.proxy_Z = tmpZ[0].asnumpy()
      assert(self.proxy_num==tmpZ[0].shape[0])
      print 'load proxy_Z from', self.proxy_Z_fn
    proxy_Z_ptmp = np.random.rand(self.proxy_batchsize, self.featdim)-0.5
    self.proxy_Z_p = proxy_Z_ptmp.astype(np.float32)
    self.proxy_Z_map = np.zeros(self.proxy_batchsize, dtype=np.int32)-1
    self.caridnum = None
    self.total_proxy_batch_epoch = 0
    self.repeat_times = repeat_times
    self.do_reset()

  def __iter__(self):
    return self

  def reset(self):
    self.cur_batch = 0        
    self.batch_carids = []
    self.batch_infos = []
    pass

  def proxy_updateset(self, proxy_Z_p_new):
    num = np.sum(self.proxy_Z_map>-1)
    p_Z = proxy_Z_p_new.asnumpy()
    self.proxy_Z_p[:] = p_Z
    for i in xrange(num):
      carid = self.proxy_Z_map[i]
      self.proxy_Z[carid] = p_Z[i]
    savename = self.proxy_Z_fn 
    mx.nd.save(savename, [mx.nd.array(self.proxy_Z)])
#    a = self.proxy_Z
#    a = p_Z
    print 'save proxy_Z into file', savename#, a#, a[a<0], a[a>0]
    pass

  def do_reset(self):
    self.cur_batch = 0        
    self.batch_carids = []
    self.batch_infos = []
    if self.total_proxy_batch_epoch == 0 \
       or self.cur_proxy_batch == self.num_proxy_batch \
       or (self.num_proxy_batch_max > 0.0 \
       and self.cur_proxy_batch > self.num_proxy_batch * self.num_proxy_batch_max):
      self.cur_proxy_batch = 0 
      self.big_epoch += 1
      self.rndidx_list = np.random.permutation(self.datalen)

    self.proxy_datalist = []
    carids = {}
    self.proxy_Z_map[:] = -1
    prndidxs = np.random.permutation(self.proxy_batchsize)
    for i in xrange(self.proxy_batchsize):
      pidx = prndidxs[i]
      pxyi = self.cur_proxy_batch * self.proxy_batchsize + pidx
      idx = self.rndidx_list[pxyi]
      onedata = self.datalist[idx] 
      parts = onedata.split(',')
      path = parts[0]
      son = parts[1]
      carid = parts[2]
      if not carids.has_key(carid):
        carids[carid] = len(carids)
      ori_id = int(carid)
      proxyid = carids[carid] 
      self.proxy_Z_p[proxyid] = self.proxy_Z[ori_id]
      self.proxy_Z_map[proxyid] = ori_id
      proxy_str = '%s,%s,%s,%s'%(path, son, carid, str(proxyid))
      self.proxy_datalist.append(proxy_str)

    self.caridnum = len(carids)
    print 'got another proxy batch to train(%d/%d/%d, %d/%d) [big_epoch=%d]...'%(\
         self.caridnum, self.proxy_batchsize, self.datalen, self.cur_proxy_batch+1,\
         self.num_proxy_batch, self.big_epoch)

    self.total_proxy_batch_epoch += 1
    if self.total_proxy_batch_epoch%self.repeat_times==0:
      self.cur_proxy_batch += 1
#    print self.proxy_Z_p, self.proxy_Z_p.sum()
    return self.caridnum, self.proxy_Z_p

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label


  def next(self):
    if self.cur_batch < self.num_batches:
#      datas, labels, carids, infos = dg.get_data_label_proxy_batch_mxnet(self._provide_data, self._provide_label, self.proxy_datalist, self.cur_batch) 
      datas, labels, carids, infos = dg.get_data_label_proxy_batch_mxnet_threads(self._provide_data, self.datas_batch, self._provide_label, self.labels_batch, self.proxy_datalist, self.cur_batch, self.caridnum) 
      self.batch_carids = carids
      self.batch_infos = infos
      self.cur_batch += 1
      return mx.io.DataBatch(datas, labels)
    else:
      raise StopIteration




if __name__=='__main__':
  print 'testing DataIter.py...'
  num_batches = 10
  pair_part1_shape = (32, 3, 128, 128)
  pair_part2_shape = (32, 3, 128, 128)
  label_shape = (pair_part1_shape[0],)
  data_iter = CarReID_Iter(['part1_data', 'part2_data'], [pair_part1_shape, pair_part2_shape],
                      ['label'], [label_shape], get_pairs_data_label,
                      num_batches)
  
  for d in data_iter:
    dks = d.data.keys()
    lks = d.label.keys()
    print dks[0], ':', d.data[dks[0]].asnumpy().shape, '   ', lks[0], ':', d.label[lks[0]].asnumpy().shape






