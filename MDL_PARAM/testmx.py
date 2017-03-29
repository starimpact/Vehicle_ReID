
import mxnet as mx
import numpy as np


def test():
  ashape = (1, 200)
  bshape = (4, 200)
  a = mx.sym.Variable('a')
  b = mx.sym.Variable('b')
#  a1 = mx.sym.broadcast_to(a, shape=bshape)
  c = mx.sym.broadcast_minus(a, b)
  c = mx.sym.sum_axis(c, axis=1)
  bs = mx.sym.SliceChannel(b, axis=0, num_outputs=4)
#  cs = mx.sym.Group([c, bs[0]])
  cs = mx.sym.minimum(c, -10000)#mx.sym.sum(c)
  c_exe = cs.bind(ctx=mx.cpu(), args={'a':mx.nd.ones(ashape), 'b':mx.nd.ones(bshape)})
  args = c_exe.arg_dict
  args['a'][:] = mx.nd.array(np.random.rand(*ashape))
 
  c_exe.forward()
  print c_exe.outputs[0].asnumpy()
#  print c_exe.outputs[1].asnumpy().shape


if __name__=='__main__':
  test()




