
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


def test2():
  ashape = (2, 20)
  bshape = (2, 20)
  a = mx.sym.Variable('a')
  b = mx.sym.Variable('b')
  a1 = a*2
  d = b + a1
  c = d[0]
  cl = mx.sym.MakeLoss(c)
  c_exe = cl.simple_bind(ctx=mx.gpu(1), a=ashape, b=bshape)
  args = c_exe.arg_dict
  grad = c_exe.grad_dict
  args['a'][:] = mx.nd.array(np.random.rand(*ashape), ctx=mx.cpu())
 
  c_exe.forward(is_train=True)
  c_exe.backward()
  print grad['a'].asnumpy()
  print c_exe.outputs[0].asnumpy()


#class Cast_Type(mx.operator.CustomOp):
#  def forward(self, is_train, req, in_data, out_data, aux):
#    self.assign
#  def __init__(self):
#    super(Cast_Type, self).__init__(need_top_grad=True)
#
#  def list_arguments(self):
#    return ['data']
#
#  def list_outputs(self):
#    return ['output']
#
#  def infer_shape(self, inshape):


#def init_args(args):
#  initializer = mx.init.Normal()
#  for key in args:
#    arr = args[key]
#    if key.endswith('_weight'):
#      initializer(mx.init.InitDesc(key), arr)
##      initializer(key, arr)
#    if key.endswith('_bias'):
#      arr[:] = 0.0
#    if key.endswith('_gamma'):
#      arr[:] = 1.0
#    if key.endswith('_beta'):
#      arr[:] = 0.0
#    if key.endswith('_init_c'):
#      arr[:] = 0.0
#    if key.endswith('_init_h'):
#      arr[:] = 0.0


featdim = 2
def create_reid4_net(batch_size, proxy_num):
  one_feat = mx.sym.Variable('data')
  one_proxy_yM = mx.sym.Variable('proxy_yM')
  proxy_Z = mx.sym.Variable(name='proxy_Z_weight',
                       shape=(proxy_num, featdim), dtype=np.float32)
  one_proxy_ZM = mx.sym.Variable('proxy_ZM')

  tzM = mx.sym.Reshape(one_proxy_ZM, shape=(-1,))
  z = mx.sym.broadcast_minus(one_feat, proxy_Z)
  z = mx.sym.abs(z)
  z = mx.sym.sum_axis(z, axis=1)

  tyM = mx.sym.Reshape(one_proxy_yM, shape=(-1, 1)) 
  one_proxy_y = mx.sym.broadcast_mul(tyM, proxy_Z)
  one_proxy_y = mx.sym.sum_axis(one_proxy_y, axis=0)
  one_feat = mx.sym.Reshape(one_feat, shape=(-1,))
  y = one_feat - one_proxy_y
  y = mx.sym.abs(y)
  y = mx.sym.sum(y)

  z_y = mx.sym.broadcast_minus(z, y)
  one_proxy_nca = mx.sym.sum(z_y * tzM)

  reid_net = one_proxy_nca 
  reid_net = mx.sym.MakeLoss(reid_net, name='proxy_nca_loss')

  return reid_net

def get_params(sym, ctx, data_shape, label_shape):
  arg_names = sym.list_arguments()
  print arg_names
  arg_shapes, _, _ = sym.infer_shape(data=data_shape, 
                                      **label_shape)

  arg_params = {}
  update_params = {}
  grad_req = {}
  grad_params = {}
  for name, shape in zip(arg_names, arg_shapes):
    arg_params[name] = mx.nd.zeros(shape, ctx)
    if name.endswith('weight') or name.endswith('bias') or \
        name.endswith('gamma') or name.endswith('beta'):
      update_params[name] = arg_params[name]
      grad_req[name] = 'write'
      grad_params[name] = mx.nd.zeros(shape, ctx)
    elif True and name=='proxy_yM':
      grad_req[name] = 'write'
      grad_params[name] = mx.nd.zeros(shape, ctx)

  return arg_params, grad_req, grad_params

def test4():
  ctx = mx.gpu(0)
  batchsize = 1
  proxy_num = 10
  yM_shape = (batchsize, proxy_num) 
  ZM_shape = (batchsize, proxy_num) 
  data_shape = (batchsize, featdim)
  label_shape = {'proxy_yM':yM_shape, 'proxy_ZM':ZM_shape}

  inputypes = {'data':np.float32, 'proxy_yM':np.float32, 
               'proxy_ZM':np.float32}
  
  net = create_reid4_net(batchsize, proxy_num) 
  argt, _, auxt = net.infer_type(**inputypes)
  arg_params, grad_req, grad_params = get_params(net, ctx, data_shape, label_shape)
  executor = net.bind(ctx, arg_params, args_grad=grad_params, grad_req=grad_req)

  grad_dict = executor.grad_dict
  arg_dict = executor.arg_dict 
  arg_dict['data'][:] = np.asarray(range(np.prod(data_shape)), dtype=np.float32).reshape(data_shape)/np.prod(data_shape)
  arg_dict['proxy_yM'][:] = 0
  arg_dict['proxy_yM'][0, 1] = 1
  arg_dict['proxy_ZM'][:] = 1
  arg_dict['proxy_ZM'][0, 1] = 0 
  zshape = arg_dict['proxy_Z_weight'].shape
  zdata = np.asarray(range(np.prod(zshape)), dtype=np.float32)/np.prod(zshape)
  arg_dict['proxy_Z_weight'][:] = zdata.reshape(zshape)


  executor.forward(True)
  executor.backward()

  print 'output:', executor.outputs[0].asnumpy().mean()
  print 'data mean:', arg_dict['data'].asnumpy().mean()
  print 'Z_weight:', arg_dict['proxy_Z_weight'].asnumpy().mean() 
  print 'grad Z_weight:', grad_dict['proxy_Z_weight'].asnumpy().mean()



if __name__=='__main__':
  test4()
  test4()




