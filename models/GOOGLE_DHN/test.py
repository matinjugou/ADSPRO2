import numpy as np
import scipy as sp
import sys
import caffe
from multiprocessing import Pool 
from collections import Counter
import random

params = dict(model_file="./deploy.prototxt",
                      pretrained_model="./final_bvlc_googlenet_iter_3000.caffemodel",
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
print "start predict"
codes = net.predict(images, oversample = False)
print "predict finished"
[database_names.append(line.strip().split(" ")[0]) for line in database]
[database_code.append(c) for c in codes]
[database_labels.append(l) for l in labels]

codebook = []
for i in range(len(database_names)):
	codebook.append(dict(name = database_names[i], label = database_labels[i], code = database_code[i]))
print "codebook created"
searchcode = net.predict([caffe.io.load_image(sys.argv[1])], oversample = False)[0]
database_searchcode = np.array(database_code);

ans = open('./result', 'w')
print "sim calcu started"
sim = np.dot(searchcode, database_searchcode.T)
print "sim calcu finished"
ids = np.argsort(-sim, axis = 0);
print "sim sort finished"

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
