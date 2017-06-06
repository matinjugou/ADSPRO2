# hash-caffe

This is a caffe repository for learning to hash. We fork the repository from [hash-caffe](https://github.com/thuml/hash-caffe) and make our modifications. The main modifications are listed as follow:

- Change the deep learning network model from Alexnet(2012) to Googlenet(2014)
- Change the weight which quantization_loss weighs to get a better performance

Data Preparation
---------------
In `data/adspro2/train.txt`, we give an example to show how to prepare training data. In `data/nus_wide/parallel/`, the list of testing and database images are splitted to 12 parts, which could be processed parallelly when predicting.

Training Model
---------------

In `models/GOOGLE_DHN/`, we give an example to show how to train hash model. In this model, we use pairwise loss and quantization loss as loss functions.

The [bvlc\_googlenet\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) is used as the pre-trained model. If the NUS\_WIDE dataset and pre-trained caffemodel is prepared, the example can be run with the following command:
```
"./build/tools/caffe train -solver models/GOOGLE_DHN/solver.prototxt -weights models/GOOGLE_DHN/bvlc_googlenet.caffemodel"
```

Parameter Tuning
---------------
In pairwise loss layer and quantization loss layer, parameter `loss_weight` can be tuned to give them different weights.

Predicting
---------------
In `models/DHN/predict/predict_parallel.py`, we give an example to show how to evaluate the trained hash model.