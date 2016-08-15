import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, filters
import sys, time, argparse
import shapely.geometry
import util

# COCO API
coco_root = '../../coco'  # modify to point to your COCO installation
sys.path.insert(0, coco_root + '/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask

# CAFFE
caffe_root = '..'
sys.path.insert(0, caffe_root + '/python')
import caffe

# PARAMS
tags, tag2ID = util.loadTags(caffe_root + '/models/COCO/catName.txt')
imgScale = 224
topBlobName = 'loss3/classifier'
topLayerName = 'loss3/classifier'
secondTopLayerName = 'pool5/7x7_s1'
secondTopBlobName = 'pool5/7x7_s1'
outputLayerName = 'pool2/3x3_s2'
outputBlobName = 'pool2/3x3_s2'

def parseArgs():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Excitation Backprop')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device ID to use [0]',
                        default=0, type=int)
    args = parser.parse_args()
    return args

# CAFFE
def initCaffe(args):
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(caffe_root+'/models/COCO/deploy.prototxt',
                    caffe_root+'/models/COCO/GoogleNetCOCO.caffemodel',
                    caffe.TRAIN)
    return net

def doExcitationBackprop(net, img, tagName):
    # load image, rescale
    minDim = min(img.shape[:2])
    newSize = (int(img.shape[0]*imgScale/float(minDim)), int(img.shape[1]*imgScale/float(minDim)))
    imgS = transform.resize(img, newSize)

    # reshape net
    net.blobs['data'].reshape(1,3,newSize[0],newSize[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # forward pass
    net.blobs['data'].data[...] = transformer.preprocess('data', imgS)
    out = net.forward(end = topLayerName)

    # switch to the excitation backprop mode
    caffe.set_mode_eb_gpu() 

    tagID = tag2ID[tagName]
    net.blobs[topBlobName].diff[0][...] = 0
    net.blobs[topBlobName].diff[0][tagID] = np.exp(net.blobs[topBlobName].data[0][tagID].copy())
    net.blobs[topBlobName].diff[0][tagID] /= net.blobs[topBlobName].diff[0][tagID].sum()

    # invert the top layer weights
    net.params[topLayerName][0].data[...] *= -1
    out = net.backward(start = topLayerName, end = secondTopLayerName)
    buff = net.blobs[secondTopBlobName].diff.copy()

    # invert back
    net.params[topLayerName][0].data[...] *= -1 
    out = net.backward(start = topLayerName, end = secondTopLayerName)

    # compute the contrastive signal
    net.blobs[secondTopBlobName].diff[...] -= buff

    # get attention map
    out = net.backward(start = secondTopLayerName, end = outputLayerName)
    attMap = np.maximum(net.blobs[outputBlobName].diff[0].sum(0), 0)

    # resize back to original image size
    attMap = transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'nearest')
    return attMap


def evalPointingGame(cocoAnn, cat, caffeNet, imgDir):
    imgIds  = cocoAnn.getImgIds(catIds=cat['id'])
    imgList = cocoAnn.loadImgs(ids=imgIds)
    hit  = 0
    miss = 0
    t0 = time.time()
    for I in imgList:
        # run EB on img, get max location on attMap
        imgName = imgDir + I['file_name']
        img     = caffe.io.load_image(imgName)
        attMap  = doExcitationBackprop(caffeNet, img, cat['name'])
        if 1:
            # naively take argmax
            maxSub = np.unravel_index(np.argmax(attMap), attMap.shape)
        else:
            # take center of max locations
            maxAtt = np.max(attMap)
            maxInd = np.where(attMap == maxAtt)
            maxSub = (np.mean(maxInd[0]), np.mean(maxInd[1]))

        # load annotations
        annList = cocoAnn.loadAnns(cocoAnn.getAnnIds(imgIds=I['id'], catIds=cat['id']))

        # hit/miss?
        isHit = 0
        for ann in annList:
            # create a radius-15 circle around max location and see if it 
            # intersects with segmentation mask
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    polyPts = np.array(seg).reshape((len(seg)/2, 2))
                    poly    = shapely.geometry.Polygon(polyPts)
                    circ    = shapely.geometry.Point(maxSub[::-1]).buffer(15)
                    isHit  += poly.intersects(circ)
            else:
                # RLE
                if type(ann['segmentation']['counts']) == list:
                    rle = mask.frPyObjects([ann['segmentation']], I['height'], I['width'])
                else:
                    rle = [ann['segmentation']]
                m = mask.decode(rle)
                m = m[:, :, 0]
                ind  = np.where(m>0)
                mp   = shapely.geometry.MultiPoint(zip(ind[0], ind[1]))
                circ = shapely.geometry.Point(maxSub).buffer(15)
                isHit += circ.intersects(mp)

            if isHit:
                break

        if isHit: 
            hit += 1
        else:
            miss += 1
        accuracy = (hit+0.0)/(hit+miss)

        if time.time() - t0 > 10: 
            print cat['name'], ': Hit =', hit, 'Miss =', miss, ' Acc =', accuracy
            t0 = time.time()

    return accuracy


if __name__ == '__main__':
    args = parseArgs()
    print args

    # load COCO val2014
    imset   = 'val2014'
    imgDir  = '%s/images/%s/'%(coco_root, imset)
    annFile = '%s/annotations/instances_%s.json'%(coco_root, imset)
    cocoAnn = COCO(annFile)
    cocoAnn.info()
    catIds  = cocoAnn.getCatIds()
    catList = cocoAnn.loadCats(catIds)

    # init caffe
    caffeNet = initCaffe(args)

    # get per-category accuracies
    accuracy = []
    for cat in catList:
        catAcc = evalPointingGame(cocoAnn, cat, caffeNet, imgDir)
        print cat['name'], ' Acc =', catAcc 
        accuracy.append(catAcc)

    # report
    for c in range(len(catList)):
        print catList[c]['name'], ': Acc =', accuracy[c]
    print 'mean Acc =', np.mean(accuracy)

