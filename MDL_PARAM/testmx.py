
import mxnet as mx
import numpy as np


def test():
  ashape = (128*32, 20000)
  bshape = (128*32, 20000)
  a = mx.sym.Variable('a')
  b = mx.sym.Variable('b')
  idx = mx.sym.Variable('idx')
  a1 = a*2
#  a1 = mx.sym.broadcast_to(a, shape=bshape)
  b = mx.sym.slice_axis(b, axis=0, begin=0, end=1)
  c = mx.sym.broadcast_minus(a, b)
#  c = mx.sym.sum_axis(c, axis=1)
#  bs = mx.sym.SliceChannel(b, axis=0, num_outputs=4)
  cl = mx.sym.MakeLoss(c)
  cs = mx.sym.Group([cl, a1])
#  cs = mx.sym.minimum(c, -10000)#mx.sym.sum(c)
#  c_exe = cs.bind(ctx=mx.cpu(), args={'a':mx.nd.ones(ashape), 'b':mx.nd.ones(bshape)})
  c_exe = cs.simple_bind(ctx=mx.gpu(1), a=ashape, b=bshape)
  args = c_exe.arg_dict
  grad = c_exe.grad_dict
  args['a'][:] = mx.nd.array(np.random.rand(*ashape), ctx=mx.cpu())
 
  c_exe.forward(is_train=True)
  lossa = mx.nd.ones(ashape, ctx=mx.gpu(1))*1 
  lossb = mx.nd.ones(ashape, ctx=mx.gpu(1))*1
  c_exe.backward(out_grads=[lossa, lossb])
  print grad['a'].asnumpy()
  print c_exe.outputs[0].asnumpy().shape
  print c_exe.outputs[1].asnumpy().shape


if __name__=='__main__':
  test()




