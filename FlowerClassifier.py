#Flower Classifier
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.reset_default_graph()

IMG_SIZE = 75
Alpha = 1e-3

MODEL_NAME = 'flowers--{}--{}.model'.format(Alpha, 'flowerConvInception') #Save Name

DAISY_DIR = r'C:\Users\david\Desktop\Flowers\flowers\daisy'
DANDELION_DIR = r'C:\Users\david\Desktop\Flowers\flowers\dandelion'
ROSE_DIR = r'C:\Users\david\Desktop\Flowers\flowers\rose'
SUNFLOWER_DIR = r'C:\Users\david\Desktop\Flowers\flowers\sunflower'
TULIP_DIR = r'C:\Users\david\Desktop\Flowers\flowers\tulip'

def label_img(img, word_label): #Label Images 
    if word_label == 'daisy': 
        return [0,0,0,0,1]
    elif word_label == 'dandelion':
        return [0,0,0,1,0]
    elif word_label == 'rose':
        return [0,0,1,0,0]
    elif word_label == 'sunflower':
        return [0,1,0,0,0]
    elif word_label == 'tulip':
        return [1,0,0,0,0]

def create_data():
    data = []
    labels = []
    for img in tqdm(os.listdir(DAISY_DIR)):
        label = label_img(img, 'daisy')
        path = os.path.join(DAISY_DIR,img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data.append(np.array(img))
        labels.append(np.array(label))
        
    for img in tqdm(os.listdir(DANDELION_DIR)):
        label = label_img(img, 'dandelion')
        path = os.path.join(DANDELION_DIR,img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data.append(np.array(img))
        labels.append(np.array(label))
        
    for img in tqdm(os.listdir(ROSE_DIR)):
        label = label_img(img, 'rose')
        path = os.path.join(ROSE_DIR,img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data.append(np.array(img))
        labels.append(np.array(label))
        
    for img in tqdm(os.listdir(SUNFLOWER_DIR)):
        label = label_img(img, 'sunflower')
        path = os.path.join(SUNFLOWER_DIR,img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data.append(np.array(img))
        labels.append(np.array(label))
        
    for img in tqdm(os.listdir(TULIP_DIR)):
        label = label_img(img, 'tulip')
        path = os.path.join(TULIP_DIR,img)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data.append(np.array(img))
        labels.append(np.array(label))
    
    np.save('data.npy', data)
    np.save('labels.npy', labels)

    
    return data, labels

#featureset, labelset = create_data() 

featureset = np.load('data.npy')
labelset = np.load('labels.npy')

print (len(featureset))
print(featureset[0].shape)

X_train, X_test, Y_train, Y_test = train_test_split(featureset, labelset)

net = input_data(shape=[None, IMG_SIZE, IMG_SIZE,3], name = 'input')

net = conv_2d(net, 32, 3, activation = 'relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 3, activation = 'relu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 128, 5, activation = 'relu')
net = max_pool_2d(net, 2, strides = 2)
net = conv_2d(net, 256, 3, activation = 'relu')
net = conv_2d(net, 384, 3, activation = 'relu')

#Inception Module (v3- replacing the 5x5 filter with 2 3x3s)

path1a = conv_2d(net, 72, 1, activation = 'relu')
path1a = conv_2d(path1a, 80, 3, activation = 'relu')
path1a = conv_2d(path1a, 92, 3, strides = 2, padding = 'valid',activation = 'relu')
path2a = conv_2d(net, 96, 1, activation = 'relu')
path2a = conv_2d(path2a, 128, 3, strides = 2, padding = 'valid',activation = 'relu')
path4a = max_pool_2d(net, 3, strides = 2, padding = 'valid')
path4a = conv_2d(path4a, 128 ,1, activation = 'relu')

net= merge([path1a, path2a, path4a], 'concat', axis = 3)

path1b = conv_2d(net, 92, 1, activation = 'relu')
path1b = conv_2d(path1b, 96, 3, activation = 'relu')
path1b = conv_2d(path1b, 102, 3, activation = 'relu')
path2b = conv_2d(net, 128, 1, activation = 'relu')
path2b = conv_2d(path2b, 130, 3, activation = 'relu')
path3b = conv_2d(net, 84, 3, activation = 'relu')
path4b = max_pool_2d(net, 2, strides = 1)
path4b = conv_2d(path4b, 92 ,1, activation = 'relu')

net= merge([path1b, path2b, path3b, path4b], 'concat', axis = 3)

path1c = conv_2d(net, 128, 1, activation = 'relu')
path1c = conv_2d(path1c, 136, 3, activation = 'relu')
path1c = conv_2d(path1c, 154, 3, activation = 'relu')
path2c = conv_2d(net, 150, 1, activation = 'relu')
path2c = conv_2d(path2c, 162, 3, activation = 'relu')
path3c = conv_2d(net, 100, 1, activation = 'relu')
path4c = max_pool_2d(net, 2, strides = 1)
path4c = conv_2d(path4c, 128 ,1, activation = 'relu')

net= merge([path1c, path2c, path3c, path4c], 'concat', axis = 3)

path1d = conv_2d(net, 156, 1, activation = 'relu')
path1d = conv_2d(path1d, 164, 3, activation = 'relu')
path1d = conv_2d(path1d, 192, 3, activation = 'relu')
path2d = conv_2d(net, 196, 1, activation = 'relu')
path2d = conv_2d(path2d, 225, 3, activation = 'relu')
path3d = conv_2d(net, 128, 1, activation = 'relu')
path4d = max_pool_2d(net, 2, strides = 1)
path4d = conv_2d(path4d, 156 ,1, activation = 'relu')

net= merge([path1d, path2d, path3d, path4d], 'concat', axis = 3)

net = max_pool_2d(net, 2, strides = 2)

net = flatten(net)
net = fully_connected(net, 1024, activation = 'relu')
net = dropout(net, .8)
net = fully_connected(net, 256, activation = 'relu')
net = fully_connected(net, 5, activation = 'softmax')

net = regression(net, optimizer='adam', learning_rate = Alpha,loss ='categorical_crossentropy', name = 'targets')

model = tflearn.DNN(net, tensorboard_dir = 'log')

if os.path.exists('{}.meta'.format(MODEL_NAME)): #Dont start from nothing!
    model.load(MODEL_NAME)
    print("loaded!")

model.fit({'input':X_train},{'targets':Y_train}, n_epoch = 1, validation_set = ({'input':X_test},{'targets':Y_test}),
      show_metric = True, snapshot_step = 500, run_id = MODEL_NAME)

model.save(MODEL_NAME)
