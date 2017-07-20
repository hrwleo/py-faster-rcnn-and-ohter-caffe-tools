#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import pickle
import cv2

caffe_root = '../'  

deployPrototxt =  '/home/hrw/caffe/extractFeature/deploy.prototxt'
modelFile = '/home/hrw/caffe/extractFeature/_iter_3112800.caffemodel'
meanFile = '/home/hrw/caffe/extractFeature/mean.npy'
#imageListFile = '/home/hrw/caffe/extractFeature/val.txt'
#imageBasePath = '/home/hrw/caffe/extractFeature/test_224'
#resultFile = 'PredictResult.txt'

#网络初始化
def initilize():
    print 'initilize ... '
    #sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net

#取出网络中的params和net.blobs的中的数据
def getNetDetails(image, net):
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(meanFile ).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  
    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  
    # the reference model has channels in BGR order instead of RGB
    # set net to batch size of 50
    net.blobs['data'].reshape(1,3,224,224)

    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    out = net.forward()
    
    #网络提取conv3_2的卷积核
    #filters = net.params['conv1_1'][0].data
    #with open('FirstLayerFilter.pickle','wb') as f:
       #pickle.dump(filters,f)
   # vis_square(filters.transpose(0, 2, 3, 1))
    #conv3_2的特征图
    feat = net.blobs['conv3_2'].data[0, :]
    with open('FirstLayerOutput.pickle','wb') as f:
       pickle.dump(feat,f)
    vis_square(feat,padval=1)
    pool = net.blobs['pool3'].data[0,:36]
    with open('pool1.pickle','wb') as f:
       pickle.dump(pool,f)
    vis_square(pool,padval=1)

    feat4 = net.blobs['conv4_3'].data[0, :]
    with open('FirstLayerOutput.pickle','wb') as f:
       pickle.dump(feat4,f)
    vis_square(feat4,padval=1)
    
    pool4 = net.blobs['pool4'].data[0,:36]
    with open('pool1.pickle','wb') as f:
       pickle.dump(pool4,f)
    vis_square(pool4,padval=1)
    
    feat5 = net.blobs['conv5_3'].data[0, :]
    with open('FirstLayerOutput.pickle','wb') as f:
       pickle.dump(feat5,f)
    vis_square(feat5,padval=1)
    
    pool5 = net.blobs['pool5'].data[0,:36]
    with open('pool1.pickle','wb') as f:
       pickle.dump(pool5,f)
    vis_square(pool5,padval=1)

# 此处将卷积图和进行显示，
def vis_square(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()
    
    #让合成图为方
    n = int(np.ceil(np.sqrt(data.shape[0])))
    print(n)
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #合并卷积图到一个图像中
    
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.imshow(data)
    plt.show()
    print("where is the pic")

if __name__ == "__main__":
    net = initilize()
    testimage = 'predict.tif'
    getNetDetails(testimage, net)
