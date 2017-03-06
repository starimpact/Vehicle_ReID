# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:37:15 2016

@author: mingzhang

PVA NET
"""


import numpy as np
import mxnet as mx
import time
import cPickle
import custom_layers


def BN_MP_Conv(inputdata, kernelsize, numfilter, conv_params, 
               poolkernel=(2, 2), poolstride=(2, 2), pooltype='max', act_type='relu', 
               do_bn=True, bn_params=None, name=''):
    if numfilter > 0:
        pady = np.int(kernelsize[0] / 2)
        padx = np.int(kernelsize[1] / 2)
        weight = mx.sym.Variable(name + '_conv_weight')
        bias = mx.sym.Variable(name + '_conv_bias')
        if conv_params is not None:
            weight = conv_params['weight']
            bias = conv_params['bias']
        pad = (pady, padx)
        conv = mx.sym.Convolution(data=inputdata, kernel=kernelsize,\
                num_filter=numfilter, pad=pad, weight=weight, bias=bias, name=name+'_conv')
    else:
        conv = inputdata
    if do_bn:
        gamma = mx.sym.Variable(name + '_bn_gamma') 
        beta = mx.sym.Variable(name + '_bn_beta') 
        moving_mean = mx.sym.Variable(name + '_bn_mov_mean')
        moving_var = mx.sym.Variable(name + '_bn_mov_var')
        if bn_params is not None:
            gamma = bn_params['gamma'] 
            beta = bn_params['beta'] 
            moving_mean = bn_params['movingmean']
            moving_var = bn_params['movingvar']
        conv = mx.sym.BatchNorm(data=conv, gamma=gamma, beta=beta,
#                                moving_mean=moving_mean, moving_var=moving_var,
                                fix_gamma=False, name=name + '_bn')
    if poolkernel[0] > 1 or poolkernel[1] > 1:
        pool = mx.sym.Pooling(data=conv, pool_type=pooltype, kernel=poolkernel, stride=poolstride, name=name+'_pool')
    else:
        pool=conv
    if act_type=='':
        relu = pool
    else:
        relu = mx.sym.Activation(data=pool, act_type=act_type, name=name+'_'+act_type)

    return relu


def CreateModel_Color_pre(data1):
    print 'CreateModel Softmax Color...'
    usebn = True

    layers_desc = (
                  ('COV', ((3, 3), 32), ((1, 1), (1, 1), 'max'), usebn),
                  ('COV', ((3, 3), 32), ((2, 2), (2, 2), 'max'), usebn),
                  ('COV', ((3, 3), 64), ((1, 1), (1, 1), 'max'), usebn),
                  ('COV', ((3, 3), 64), ((2, 2), (2, 2), 'max'), usebn),
                  ('COV', ((3, 3), 128), ((1, 1), (1, 1), 'max'), usebn), 
                  ('COV', ((3, 3), 128), ((2, 2), (2, 2), 'max'), usebn), 
                  ('COV', ((3, 3), 256), ((1, 1), (1, 1), 'max'), usebn), 
                  ('COV', ((3, 3), 256), ((2, 2), (2, 2), 'max'), usebn), 
                  ('COV', ((3, 3), 512), ((1, 1), (1, 1), 'max'), usebn), 
                  ('COV', ((3, 3), 512), ((2, 2), (2, 2), 'max'), usebn), 
                  ('COV', ((3, 3), 1024), ((1, 1), (1, 1), 'max'), usebn), 
                  ('COV', ((3, 3), 1024), ((2, 2), (2, 2), 'max'), usebn), 
                  ('COV', ((3, 3), 2048), ((1, 1), (1, 1), 'max'), usebn), 
                  ('COV', ((3, 3), 2048), ((2, 2), (2, 2), 'max'), usebn), 
                  ('COV', ((3, 3), 0), ((2, 2), (2, 2), 'max'), usebn), 
                  )

    layernum = len(layers_desc)

    layer_params = [] 
    for i in xrange(layernum):
        now_desc = layers_desc[i]
        if now_desc[0] == 'COV':
            layerp = [{'weight':mx.sym.Variable('conv_' + str(i) + '_weight'), 
                       'bias':mx.sym.Variable('conv_' + str(i) + '_bias')}]
            if now_desc[-1]:
                layerp += [{'gamma':mx.sym.Variable('bn_' + str(i) + '_gamma'),
                            'beta':mx.sym.Variable('bn_' + str(i) + '_beta'),
                            'movingmean':mx.sym.Variable('bn_' + str(i) + '_movingmean'),
                            'movingvar':mx.sym.Variable('bn_' + str(i) + '_movingvar')}]

        layer_params.append(layerp)
     
    # input 1
    datapre = data1
    convlayers1 = []
    nkerns1 = []
    numfilter_pre = 0
    for i in xrange(layernum):
        now_desc = layers_desc[i]
        if now_desc[0] == 'COV':
            layerp = layer_params[i]
            conv_p = now_desc[1]
            pool_p = now_desc[2]
            layernow = BN_MP_Conv(inputdata=datapre, kernelsize=conv_p[0], numfilter=conv_p[1], conv_params=layerp[0], 
                                  poolkernel=pool_p[0], poolstride=pool_p[1], pooltype=pool_p[2], act_type='relu', 
                                  bn_params=layerp[1], name='PART1_COV_' + str(i))
            numfilter_pre = conv_p[1]
        datapre = layernow
        nkerns1.append(numfilter_pre)
        convlayers1.append(layernow)
    flatten1 = mx.sym.Flatten(data=convlayers1[-1], name="flatten1") 
    drop1 = mx.sym.Dropout(data=flatten1, p=0.5, name='dropout1')
    reid_fc1 = mx.sym.FullyConnected(data=drop1, num_hidden=512, name="fc1") 
    reid_act = mx.sym.Activation(data=reid_fc1, act_type='relu', name='fc1_relu')
    
    return reid_act


def CreateModel_Color(ctxdev, batchsize, imgsize, clsnum):
    imgh, imgw = imgsize
    
    imgchnum = 3
    
    data1 = mx.sym.Variable('data')
    reid_feature = CreateModel_Color_pre(data1)
    reid_cls = mx.sym.FullyConnected(data=reid_feature, num_hidden=clsnum, name="fc_cls") 
    label = mx.sym.Variable('label')
    reid_net = mx.sym.SoftmaxOutput(data=reid_cls, label=label, name='cls') 
   
    if False:
        reid_net_exec = reid_net.simple_bind(ctx=ctxdev, data=(batchsize, imgchnum, imgh, imgw), 
                                             label=batchsize, grad_req='write')
        reid_net_args = reid_net_exec.arg_dict
        reid_net_aux = reid_net_exec.aux_dict
        reid_net_grads = reid_net_exec.grad_dict
        
        print 'args, grads, length:', len(reid_net_args), len(reid_net_grads)
        
        print 'reid_net_args'
        for key in reid_net_args:
           print key, reid_net_args[key].asnumpy().shape
    
        print 'reid_net_grads'
        for key in reid_net_grads:
           print key, reid_net_grads[key].asnumpy().shape
    
        print 'reid_net_aux'
        for key in reid_net_aux:
           print key, reid_net_aux[key].asnumpy().shape

    #darw net
    if True:
        graph = mx.visualization.plot_network(reid_net, \
            shape={'data':(batchsize, imgchnum, imgsize[0], imgsize[1]),
                   'label':batchsize})
        graph.render('reid_net_graph')
   #test time
    if False:
        tnum = 100
        t0 = time.time()
        for i in xrange(tnum):
            reid_net_exec.forward()
            mx.nd.waitall()
        t1 = time.time()
        print "predict time rpn:%.2f ms"%((t1-t0)*1000/tnum/batchsize)

    return reid_net 


def CreateModel_Color_Split_test():
   data1 = mx.sym.Variable('part1_data')
   reid_feature = CreateModel_Color_pre(data1)
   feature1 = mx.sym.Variable('feature1_data')
   feature2 = mx.sym.Variable('feature2_data')
#   absdiff = mx.sym.sum(mx.sym.abs(feature1-feature2), axis=1)
   absdiff = mx.sym.abs(feature1-feature2)
   return reid_feature, absdiff 


if __name__=="__main__":    
    ctxdev = mx.cpu(0)
    stdsize = (256, 256)
    batchsize = 10
    myreid = CreateModel_Color(ctxdev, batchsize, stdsize)


