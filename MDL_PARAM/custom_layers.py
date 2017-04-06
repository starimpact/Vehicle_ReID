# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:22:25 2016

@author: mingzhang
"""

import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np


class Masked_Softmax(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))
#        print '***:', out_data[0].asnumpy()

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        l = in_data[1].asnumpy().ravel().astype(np.int)
        y = out_data[0].asnumpy()
        y[np.arange(l.shape[0]), l] -= 1.0
        mask = in_data[2].asnumpy()
        ywv = y*mask
        self.assign(in_grad[0], req[0], mx.nd.array(ywv))


@mx.operator.register("masked_softmax")
class Masked_SoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(Masked_SoftmaxProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'label', 'mask']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        mask_shape = (in_shape[0][0], 1)
        output_shape = in_shape[0]
        return [data_shape, label_shape, mask_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return Masked_Softmax()

###############################################################################

class Masked_Layer(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
#        print len(in_data)
        x = in_data[0]
#        xx = x.asnumpy()
#        print 'xx:', xx.shape
        self.assign(out_data[0], req[0], x)
#        oo = out_data[0].asnumpy()
#        print 'oo:', oo

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        mask = in_data[1]
#        print mask.asnumpy()
#        print out_grad[0].asnumpy()
        tt = out_grad[0]*mask
#        print 'grad:', tt.shape
        self.assign(in_grad[0], req[0], tt)


@mx.operator.register("masked_layer")
class Masked_LayerProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(Masked_LayerProp, self).__init__(need_top_grad=True)
    
    def list_arguments(self):
        return ['data', 'mask']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        mask_shape = (in_shape[0][0], 1)
        output_shape = in_shape[0]
        return [data_shape, mask_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return Masked_Layer()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0], in_data[1]]

###############################################################################

class Proxy_Set(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        self.assign(out_data[0], req[0], x)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        tt = out_grad[0]
        self.assign(in_grad[0], req[0], tt)


@mx.operator.register("proxy_set")
class Proxy_SetProp(mx.operator.CustomOpProp):
    def __init__(self, proxy_num):
        super(Proxy_SetProp, self).__init__(need_top_grad=True)
        self.proxy_num = long(proxy_num)
    
    def list_arguments(self):
        return ['data', 'proxy']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, inshape):
        print inshape
        data_shape = inshape[0]
        proxy_shape = (self.proxy_num, data_shape[1])
        output_shape = data_shape
        return [data_shape, proxy_shape], [output_shape], []

    def infer_type(self, in_type):
        print 'in_type:', in_type
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return Proxy_Set()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [out_grad[0]]


###############################################################################

