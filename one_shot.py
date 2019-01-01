# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import itertools as it
from operator import itemgetter

from sklearn.model_selection import StratifiedShuffleSplit    
    
tf.logging.set_verbosity(tf.logging.INFO)
#tf.reset_default_graph()

data = pd.read_csv('./train.csv')
#df = pd.merge(data, data, on='Image')
train_data = data[~(data.Id == "new_whale")]
train_data.to_csv("./train_data_16k.csv", index=False)
le = LabelEncoder()
true_labels = le.fit_transform(np.asarray(train_data["Id"]).reshape(-1,1)).astype(np.int32)
IMG_SIZE=64
path = "./train_data/"
BATCH=500
def SiameseNet(x):
    
    features = tf.reshape(x, [BATCH, 64, 64, 1])
        
    network = tf.layers.conv2d(inputs=features,
                            filters=64,
                            kernel_size=10,
                            padding='SAME',
                            activation = tf.nn.tanh)
#kernel_initializer = tf.contrib.layers.xavier_initializer())
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                            filters=128,
                            kernel_size=7,
                            padding='SAME',
                            activation = tf.nn.elu)
#kernel_initializer = tf.contrib.layers.xavier_initializer())
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                         filters=128,
                         kernel_size=4,
                         padding='SAME',
                         activation = tf.nn.relu)
#kernel_initializer = tf.contrib.layers.xavier_initializer())
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                         filters=256,
                         kernel_size=4,
                         padding='SAME',
                         activation = tf.nn.relu)
#                         kernel_initializer = tf.contrib.layers.xavier_initializer())
    #print(network)
    network_flat = tf.reshape(network, [BATCH, 256*8*8])

    dense_nw = tf.layers.dense(inputs=network_flat, units=4096, activation=tf.nn.sigmoid)
    
    return dense_nw 
    
def model_fn(mode, features, labels):
    (img1, img2) = features
    dense1 = SiameseNet(img1)
    dense2 = SiameseNet(img2)
    l1_dist = tf.reshape(tf.abs(tf.subtract(dense1,dense2)), (BATCH,4096))
    
    y_ = tf.layers.dense(inputs=l1_dist, units=1, activation= tf.nn.sigmoid)
    y_ = tf.reshape(y_, (BATCH,1))	
    
    train_op=None
    predictions=None
    loss=None
    eval_metric_ops=None
    global_step = tf.train.get_global_step()
    if(mode==tf.estimator.ModeKeys.EVAL or 
            mode==tf.estimator.ModeKeys.TRAIN):
        loss = tf.losses.sigmoid_cross_entropy(
                    multi_class_labels=labels, logits=y_)
    if(mode==tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss,global_step = global_step)
    if(mode == tf.estimator.ModeKeys.EVAL):
        eval_metric_ops = {"absolute error": tf.metrics.mean_absolute_error(labels, y_)}
    
    predictions = {"classes": tf.round(y_), "probabilities": y_}
    
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

#def prcs(xx):
#    (x1,x2) = xx
def imgprcs(im, path):
    img = path+im
    #print(img)
    img = tf.io.read_file(img)
    oh = tf.image.extract_jpeg_shape(img)
    img = tf.image.decode_jpeg(img)
    img = tf.cond(tf.less(oh[2],3), lambda: tf.image.grayscale_to_rgb(img),
                         lambda: img)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize_images(img, [IMG_SIZE, IMG_SIZE])
    img = tf.image.per_image_standardization(img)
    return img
img1=list()
img2=list()
label=list()
def pre_processing():
    NUM_SAME_LABEL = 85634 
    imgs=train_data.Image.values
#train_data.set_index('Image')
#print(train_data)
    train_data_dict = train_data.set_index('Image').T.to_dict('dict')
#print(train_data_dict['0000e88ab.jpg'])
    perm = it.combinations_with_replacement(imgs, 2)
#print(next(perm))
    print("computing the combinations...\n")
    for i,p in enumerate(perm):
        (x1,x2) = p
        if(train_data_dict[x1]['Id']==train_data_dict[x2]['Id']): y=1.0
        else: y=0.0
        img1.append(x1)
        img2.append(x2)
        label.append(y)
    print("num_one_label:", label.count(1))
#if(i > 999): break
#print(img1)
#print(np.asarray(img1).shape)
#print(len(np.reshape(img1,(-1,1))))
#img1 = np.reshape(img1, (-1,1))
#img2 = np.reshape(img2, (-1,1))
#label = np.reshape(label,(-1,1))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=100000, test_size=5000, random_state=0)
#print(next(sss.split(img1, label)))
    train_indices, test_indices = next(sss.split(img1, label))
    return train_indices, test_indices
#train_indices, test_indices = pre_processing()
#print(train_indices)
#train_indices = list(pre_processing())
#print(train_indices)
#train_indices = np.asarray(train_indices)
'''-------------------
image1 = list(itemgetter(*train_indices)(img1))
image2 = list(itemgetter(*train_indices)(img2))
train_label = list(itemgetter(*train_indices)(label))
test_image1 = list(itemgetter(*test_indices)(img1))
test_image2 = list(itemgetter(*test_indices)(img2))
test_label = list(itemgetter(*test_indices)(label))
--------------------
'''
#return (img1[train_indices], img2[train_indices]), label[train_indices]
#for i in range(1000):
#(x1, x2) = np.random.choice(15690, 2)
#    if train_data.iloc[x1,1] == train_data.iloc[x2,1] : y = 1
#    else: y = 0
#    element1 = train_data.iloc[x1,0] #itr.get_next()
#    element2 = train_data.iloc[x2,0] #itr.get_next()
#    feat1 = imgprcs(element1)
#    feat2 = imgprcs(element2)
    #feat1, feat2 = prcs(feat)
    #feat1 = tf.reshape(tf.convert_to_tensor(feat1), (1,64,64,1))
    #feat2 = tf.reshape(tf.convert_to_tensor(feat2), (1,64,64,1))
    #y = tf.convert_to_tensor(y)
def input_fn():
    print("creating the dataset..\n")
    dataset = tf.data.Dataset.from_tensor_slices(((image1, image2), label))
    batch_dataset = dataset.batch(20)
    dataset = dataset.map(lambda itm1, itm2, l: lambda_fn(itm1,itm2,l), 4)
    itr = batch_dataset.make_one_shot_iterator()
    return itr.get_next() #next_element
def batched_input_fn():
    idx =np.random.choice(100000, BATCH, replace=False)
    x1 = list()
    x2 = list()
    yy=list()
    
    for n in idx:
        x1.append(imgprcs(image1[n],"./train_data/"))
        x2.append(imgprcs(image2[n],"./train_data/"))
        yy.append(train_label[n])
                            
    x1 = tf.reshape(tf.convert_to_tensor(x1), (BATCH, 64, 64, 1))
    x2 = tf.reshape(tf.convert_to_tensor(x2), (BATCH, 64, 64, 1))
    yy = tf.reshape(tf.convert_to_tensor(yy), (BATCH, 1))
    #print(x1, x2,yy)
    return (x1, x2), yy

def eval_input_fn():
    idx =np.random.choice(5000, BATCH, replace=False)
    x1 = list()
    x2 = list()
    yy = list()
    for n in idx:
        x1.append(imgprcs(test_image1[n],"./train_data/"))
        x2.append(imgprcs(test_image2[n],"./train_data/"))
        yy.append(test_label[n])
    x1 = tf.reshape(tf.convert_to_tensor(x1), (BATCH, 64, 64, 1))
    x2 = tf.reshape(tf.convert_to_tensor(x2), (BATCH, 64, 64, 1))
    yy = tf.reshape(tf.convert_to_tensor(yy), (BATCH, 1))
     #print(x1, x2,yy)
    return (x1, x2), yy

est = tf.estimator.Estimator(model_fn=model_fn, model_dir='./oneshot_model_dir')
#for _ in range(1):
#    est.train(input_fn = batched_input_fn, steps = 1000)
#    results = est.evaluate(input_fn=eval_input_fn,steps=100)
#    print(results)
#yy=input_fn()
#print(yy)
imgs=train_data.Image.values
print("making dict", flush=True)
train_data_dict = train_data.set_index('Image').T.to_dict('dict')
files = glob.glob("./test_data/*.jpg")
print(files)
print("making cross product", flush=True)
product1 = it.product(files, "./train_data/"+imgs)
predictions_list = list()
def pred_input_fn():
    img1=list()
    img2=list()
    for i,(x1,x2) in enumerate(product1):
        img1.append(imgprcs(x1,""))
        img2.append(imgprcs(x2,""))
        if((i+1)%BATCH == 0):
            return (img1, img2), None      
print("predicting..", flush=True)
predictions = est.predict(input_fn=pred_input_fn, predict_keys="probabilities")

product2 = it.product(files, "./train_data/"+imgs)
pred_result_top5=list()
cnt = [i+1 for i in range(7690*15697)]
probs=list()
print("decoding the predictions", flush=True)
for i,(x1,x2),pred in zip(cnt, product2,predictions):
    [p] = pred["probabilities"]
    probs.append(p)
#print(x1, x2)
#print(np.round(i/15697),"set is going on..", flush=True)
    if(i%15697 ==0):
        args = np.flip(np.argsort(probs))[:5]
        print(args, flush=True)
        imgtop5 = list(itemgetter(*args)(imgs))
        print(imgtop5, flush=True)
#result_top5.append(top5)
#print(result_top5)
#rslt = train_data.loc[train_data["Image"]==imgtop5, 'Id']
#[lbltop5] = itemgetter(*imgtop5)(train_data_dict)
        tt=list()
        for t in imgtop5:
            tt.append(train_data_dict[t]["Id"])
        print(i, x1, tt, flush=True)
        pred_result_top5.append([x1,tt])
        probs.clear()
print(pred_result_top5, flush=True)
with open("whale_Detection_submit.csv","w") as ff:
    ff.writelines(pred_result_top5)




