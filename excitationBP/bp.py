import numpy as np
import caffe

def getDeconvMap(net, param, tags, gpu = True):
    if gpu:
        caffe.set_mode_dc_gpu()
    else:
        caffe.set_mode_dc_cpu()
    topLayerName = param['topLayerName']
    outputLayerName = param['outputLayerName']
    
    if outputLayerName:
        out = net.backward(start = topLayerName, end = outputLayerName)
        attMap = net.blobs[outputLayerName].diff[0].copy()
    else:
        out = net.backward(start = topLayerName)
        attMap = net.blobs['data'].diff[0].copy()
    return attMap