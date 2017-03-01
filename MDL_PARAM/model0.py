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


def CreateModel_Color_pre():
    print 'CreateModel Color...'

    layers_desc = (
                  ('COV', ((3, 3), 32), ((3, 3), (2, 2), 'max'), True),
                  ('COV', ((3, 3), 64), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 128), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 256), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 512), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 1024), ((3, 3), (2, 2), 'max'), True), 
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
    data1 = mx.sym.Variable('part1_data')
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
    flatten1 = mx.symbol.Flatten(data=convlayers1[-1], name="flatten1") 
     
    # input 2
    data2 = mx.sym.Variable('part2_data')
    datapre = data2
    convlayers2 = []
    nkerns2 = []
    numfilter_pre = 0
    for i in xrange(layernum):
        now_desc = layers_desc[i]
        if now_desc[0] == 'COV':
            layerp = layer_params[i]
            conv_p = now_desc[1]
            pool_p = now_desc[2]
            layernow = BN_MP_Conv(inputdata=datapre, kernelsize=conv_p[0], numfilter=conv_p[1], conv_params=layerp[0], 
                                  poolkernel=pool_p[0], poolstride=pool_p[1], pooltype=pool_p[2], act_type='relu', 
                                  bn_params=layerp[1], name='PART2_COV_' + str(i))
            numfilter_pre = conv_p[1]
        datapre = layernow
        nkerns2.append(numfilter_pre)
        convlayers2.append(layernow)
    flatten2 = mx.sym.Flatten(data=convlayers2[-1], name="flatten2") 
    
    # combine
    metric_sub = mx.sym.abs(flatten1 - flatten2)
    fc_sub_score = mx.sym.FullyConnected(data=metric_sub, num_hidden=1, name='fc_sub')
    metric_mul = flatten1 * flatten2
    fc_mul_score = mx.sym.FullyConnected(data=metric_mul, num_hidden=1, name='fc_mul')
    
    hybrid_score = fc_sub_score + fc_mul_score

     
    return hybrid_score 


def CreateModel_Color(ctxdev, batchsize, imgsize):
    imgh, imgw = imgsize
    
    imgchnum = 3
    
    hybrid_score = CreateModel_Color_pre()
    label = mx.sym.Variable('label')
    cost = mx.sym.log(1.0 + mx.sym.exp(-label*hybrid_score)) 
    reid_net = mx.sym.MakeLoss(data=cost, name='reid_loss')
    
    if False:
        reid_net_exec = reid_net.simple_bind(ctx=ctxdev, part1_data=(batchsize, imgchnum, imgh, imgw), 
                                             part2_data=(batchsize, imgchnum, imgh, imgw),
                                             label=(batchsize, 1), grad_req='write')
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
            shape={'part1_data':(batchsize, imgchnum, imgsize[0], imgsize[1]),
                   'part2_data':(batchsize, imgchnum, imgsize[0], imgsize[1]),
                   'label':(batchsize, 1)})
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


def CreateModel_Color_Test(ctxdev, batchsize, imgsize):
    imgh, imgw = imgsize
    
    reid_net_test = CreateModel_Color_pre()

    if False:
        reid_net_exec = reid_net.simple_bind(ctx=ctxdev, part1_data=(batchsize, imgchnum, imgh, imgw), 
                                             part2_data=(batchsize, imgchnum, imgh, imgw), grad_req='null')
        reid_net_args = reid_net_exec.arg_dict
        reid_net_aux = reid_net_exec.aux_dict
        
        print 'args, grads, length:', len(reid_net_args), len(reid_net_aux)
        
        print 'reid_net_args'
        for key in reid_net_args:
           print key, reid_net_args[key].asnumpy().shape
   
        print 'reid_net_aux'
        for key in reid_net_aux:
           print key, reid_net_aux[key].asnumpy().shape

    return reid_net_test 



def CreateModel_Color_Split_test():
    print 'CreateModel Color Split...'

    layers_desc = (
                  ('COV', ((3, 3), 32), ((3, 3), (2, 2), 'max'), True),
                  ('COV', ((3, 3), 64), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 128), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 256), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 512), ((3, 3), (2, 2), 'max'), True), 
                  ('COV', ((3, 3), 1024), ((3, 3), (2, 2), 'max'), True), 
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
    data1 = mx.sym.Variable('part1_data')
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
    reid_feature = mx.symbol.Flatten(data=convlayers1[-1], name="flatten1") 
     
    # combine
    feature1 = mx.sym.Variable('feature1_data')
    feature2 = mx.sym.Variable('feature2_data')

    metric_sub = mx.sym.abs(feature1 - feature2)
    fc_sub_score = mx.sym.FullyConnected(data=metric_sub, num_hidden=1, name='fc_sub')
    metric_mul = feature1 * feature2
    fc_mul_score = mx.sym.FullyConnected(data=metric_mul, num_hidden=1, name='fc_mul')
    
    hybrid_score = fc_sub_score + fc_mul_score
     
    return reid_feature, hybrid_score 



if __name__=="__main__":    
    ctxdev = mx.cpu(0)
    stdsize = (128, 128)
    batchsize = 10
    myreid = CreateModel_Color(ctxdev, batchsize, stdsize)


