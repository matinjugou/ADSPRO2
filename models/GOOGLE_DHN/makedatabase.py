import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool 
from collections import Counter
import random

params = dict(model_file="./deploy.prototxt",
                      pretrained_model="./example.caffemodel",
                      image_dims=(256,256),
                      scale=255,
                      database="../../data/adspro2/train.txt",
                      batch_size=50,
                      mean_file="./ilsvrc_2012_mean.npy",
                      gpu_id=0);

caffe.set_device(params['gpu_id'])
caffe.set_mode_gpu()
model_file = params['model_file']
pretrained_model = params['pretrained_model']
scale = params['scale']
database = open(params['database'], 'r').readlines()
batch_size = params['batch_size']
dims = params['image_dims']

if 'mean_file' in params:
	mean_file = params['mean_file']
	net = caffe.Classifier(model_file,pretrained_model,channel_swap = (2,1,0), image_dims = dims, mean = np.load(mean_file).mean(1).mean(1), raw_scale = scale)
else:
	net = caffe.Classifier(model_file, pretrained_model, channel_swap = (2,1,0), image_dims = dims, raw_scale = scale)

database_code = []
database_labels = []
database_names = []

images = [caffe.io.load_image(line.strip().split(" ")[0]) for line in database]
labels = [[int(i) for i in line.strip().split(" ")[1:]] for line in database]
codes = net.predict(images, oversample = False)
[database_names.append(line.strip().split(" ")[0]) for line in database]
[database_code.append(c) for c in codes]
[database_labels.append(l) for l in labels]

codebook = []
for i in range(len(database_names)):
	codebook.append(dict(name = database_names[i], label = database_labels[i], code = database_code[i]))
database_searchcode = np.array(database_code)
database_labels = np.array(database_labels)
database_names = np.array(database_names)
np.save("./database_code.npy",database_searchcode)
np.save("./database_labels.npy",database_labels)
np.save("./database_names.npy",database_names)
