import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool 
from collections import Counter
import random

params = dict(model_file="./deploy.prototxt",
                      pretrained_model="./result.caffemodel",
                      image_dims=(256,256),
                      scale=255,
                      database="../../data/adspro2/train.txt",
                      batch_size=50,
                      mean_file="./ilsvrc_2012_mean.npy",
                      gpu_id=1);

caffe.set_device(params['gpu_id'])
caffe.set_mode_gpu()
model_file = params['model_file']
pretrained_model = params['pretrained_model']
scale = params['scale']
batch_size = params['batch_size']
dims = params['image_dims']

if 'mean_file' in params:
	mean_file = params['mean_file']
	net = caffe.Classifier(model_file,pretrained_model,channel_swap = (2,1,0), image_dims = dims, mean = np.load(mean_file).mean(1).mean(1), raw_scale = scale)
else:
	net = caffe.Classifier(model_file, pretrained_model, channel_swap = (2,1,0), image_dims = dims, raw_scale = scale)

database_searchcode = np.load("database_code.npy")
database_labels = np.load("database_labels.npy")
database_names = np.load("database_names.npy")

searchcode = net.predict([caffe.io.load_image(sys.argv[1])], oversample = False)[0]

ans = open('./result', 'w')
sim = np.dot(searchcode, database_searchcode.T)
ids = np.argsort(-sim, axis = 0);

ansset = []
items = []
for i in range(200):
	ansset.append(str(database_names[ids[i]]))
	ans.write(str(ansset[i])+"\n")
	items.append(str(database_names[ids[i]].strip().split("_")[0]))
	

count = Counter(items)
print count
ans.write("\n -----------------divide--------------\n")

newlabellist = sorted(count.items(),key=lambda d:d[1])
print newlabellist
ansset = [i for i in ansset if i.find(str(newlabellist[len(newlabellist) - 1][0])) <> -1 or random.uniform(0,10) > 8]

for i in ansset:
	ans.write(str(i) + "\n");
