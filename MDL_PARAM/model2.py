# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:37:15 2016

@author: mingzhang

PVA NET
"""

"""
Contains the definition of the Inception Resnet V2 architecture.		
As described in http://arxiv.org/abs/1602.07261.		
Inception-v4, Inception-ResNet and the Impact of Residual Connections		
on Learning		
Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi		
"""


import numpy as np
import mxnet as mx
import time
import cPickle
import custom_layers


def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), act_type="relu", mirror_attr={}, with_act=True):
  conv = mx.symbol.Convolution(
      data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
  bn = conv#mx.symbol.BatchNorm(data=conv)
  if with_act:
      act = mx.symbol.Activation(data=bn, act_type=act_type, attr=mirror_attr)
      return act
  else:
      return bn


def stem(data):
  conv1a_3_3 = ConvFactory(data=data, num_filter=32,
                           kernel=(3, 3), stride=(2, 2))
  conv2a_3_3 = ConvFactory(conv1a_3_3, 32, (3, 3))
  conv2b_3_3 = ConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1))
  maxpool3a_3_3 = mx.symbol.Pooling(
      data=conv2b_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')
  conv3b_1_1 = ConvFactory(maxpool3a_3_3, 80, (1, 1))
  conv4a_3_3 = ConvFactory(conv3b_1_1, 192, (3, 3))

  return conv4a_3_3 


def reductionA(conv4a_3_3):
  maxpool5a_3_3 = mx.symbol.Pooling(
      data=conv4a_3_3, kernel=(3, 3), stride=(2, 2), pool_type='max')

  tower_conv = ConvFactory(maxpool5a_3_3, 96, (1, 1))
  tower_conv1_0 = ConvFactory(maxpool5a_3_3, 48, (1, 1))
  tower_conv1_1 = ConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2))

  tower_conv2_0 = ConvFactory(maxpool5a_3_3, 64, (1, 1))
  tower_conv2_1 = ConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1))
  tower_conv2_2 = ConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1))

  tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
      3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
  tower_conv3_1 = ConvFactory(tower_pool3_0, 64, (1, 1))
  tower_5b_out = mx.symbol.Concat(
      *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
  return tower_5b_out 


def reductionB(net):
  tower_conv = ConvFactory(net, 384, (3, 3), stride=(2, 2))
  tower_conv1_0 = ConvFactory(net, 256, (1, 1))
  tower_conv1_1 = ConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1))
  tower_conv1_2 = ConvFactory(tower_conv1_1, 384, (3, 3), stride=(2, 2))
  tower_pool = mx.symbol.Pooling(net, kernel=(
      3, 3), stride=(2, 2), pool_type='max')
  net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])

  return net


def reductionC(net):
  tower_conv = ConvFactory(net, 256, (1, 1))
  tower_conv0_1 = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2))
  tower_conv1 = ConvFactory(net, 256, (1, 1))
  tower_conv1_1 = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2))
  tower_conv2 = ConvFactory(net, 256, (1, 1))
  tower_conv2_1 = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1))
  tower_conv2_2 = ConvFactory(tower_conv2_1, 320, (3, 3),  stride=(2, 2))
  tower_pool = mx.symbol.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type='max')
  net = mx.symbol.Concat(*[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])
  return net


def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
  tower_conv = ConvFactory(net, 32, (1, 1))
  tower_conv1_0 = ConvFactory(net, 32, (1, 1))
  tower_conv1_1 = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1))
  tower_conv2_0 = ConvFactory(net, 32, (1, 1))
  tower_conv2_1 = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1))
  tower_conv2_2 = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1))
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
  tower_out = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False)

  net += scale * tower_out
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr)
      return act
  else:
      return net


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
  tower_conv = ConvFactory(net, 192, (1, 1))
  tower_conv1_0 = ConvFactory(net, 129, (1, 1))
  tower_conv1_1 = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2))
  tower_conv1_2 = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1))
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
  tower_out = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False)
  net += scale * tower_out
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr)
      return act
  else:
      return net


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr={}):
  tower_conv = ConvFactory(net, 192, (1, 1))
  tower_conv1_0 = ConvFactory(net, 192, (1, 1))
  tower_conv1_1 = ConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1))
  tower_conv1_2 = ConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0))
  tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
  tower_out = ConvFactory(
      tower_mixed, input_num_channels, (1, 1), with_act=False)
  net += scale * tower_out
  if with_act:
      act = mx.symbol.Activation(
          data=net, act_type=act_type, attr=mirror_attr)
      return act
  else:
      return net


def repeat(inputs, repetitions, layer, *args, **kwargs):
  outputs = inputs
  for i in range(repetitions):
      outputs = layer(outputs, *args, **kwargs)
  return outputs


def get_symbol(num_classes=1000, **kwargs):
  data = mx.symbol.Variable(name='data')

  stem_net = stem(data)

  reduceA = reductionA(stem_net)

  net = repeat(reduceA, 2, block35, scale=0.17, input_num_channels=320)

  reduceB = reductionB(net)

  net = repeat(reduceB, 4, block17, scale=0.1, input_num_channels=1088)

  reduceC = reductionC(net)

  net = repeat(reduceC, 2, block8, scale=0.2, input_num_channels=2080)
  net = block8(net, with_act=False, input_num_channels=2080)

  net = ConvFactory(net, 1536, (1, 1))
  net = mx.symbol.Pooling(net, kernel=(8, 8), stride=(1, 1), pool_type='avg')
  net = mx.symbol.Flatten(net)
#  net = mx.symbol.Dropout(data=net, p=0.2)
#  net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes)
#  softmax = mx.symbol.SoftmaxOutput(data=net, name='softmax')
  return net

def draw_inception_renet_v2():
  net = get_symbol()
  #darw net
  datashape = (1, 3, 299, 299)
  graph = mx.visualization.plot_network(net, shape={'data':datashape})
  graph.render('inception_renet_v2') 


if __name__=="__main__":
  draw_inception_renet_v2()


"""
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
"""





