
import mxnet as mx
import numpy as np


def test():
  ashape = (1, 20)
  bshape = (1, 20)
  a = mx.sym.Variable('a')
  b = mx.sym.Variable('b')
  idx = mx.sym.Variable('idx')
  a1 = a*2
#  a1 = mx.sym.broadcast_to(a, shape=bshape)
  b = mx.sym.slice_axis(b, axis=0, begin=0, end=1)
  c = mx.sym.broadcast_minus(a, b)
#  c = mx.sym.sum_axis(c, axis=1)
#  bs = mx.sym.SliceChannel(b, axis=0, num_outputs=4)
  cs = mx.sym.Group([c, a1[0]])
#  cs = mx.sym.minimum(c, -10000)#mx.sym.sum(c)
  c_exe = cs.bind(ctx=mx.cpu(), args={'a':mx.nd.ones(ashape), 'b':mx.nd.ones(bshape)})
  args = c_exe.arg_dict
  args['a'][:] = mx.nd.array(np.random.rand(*ashape))
 
  c_exe.forward()
  print c_exe.outputs[0].asnumpy().shape
#  c_exe.outputs[1][:] = 0
  print c_exe.outputs[1].asnumpy()


if __name__=='__main__':
  test()




