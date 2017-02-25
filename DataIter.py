import numpy as np
import mxnet as mx
import DataGenerator as dg

datafn = '/media/data1/mzhang/data/car_ReID_for_zhangming/data.list'

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






