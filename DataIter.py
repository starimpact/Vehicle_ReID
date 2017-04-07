import numpy as np
import mxnet as mx 
import DataGenerator as dg 

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
      datas, labels = dg.get_data_label_proxy_mxnet(self._provide_data, self._provide_label, self.datalist, self.rndidx_list, self.cur_batch) 
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






