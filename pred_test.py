#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:40:12 2022

@author: bowen
"""

#%% Get breed name from one-hot enc.
from keras.layers import Dense,Dropout,Input,MaxPooling2D,ZeroPadding2D,Conv2D,Flatten,BatchNormalization,Activation
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
from keras_preprocessing.image import load_img,ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array

df1=pd.read_csv('DBI-Keras/Kdata/labels.csv')
df1.head()
img_file='DBI-Keras/Kdata/train/'
df=df1.assign(img_path=lambda x: img_file + x['id'] +'.jpg')
df.head()
df.breed.value_counts()
top_20=list(df.breed.value_counts()[0:20].index)
top_20 # Display top 20 breeds name.
df2=df[df.breed.isin(top_20)]
df2.shape # Check total amount of images in the top_20 set.

img_pixel=np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in df2['img_path'].values.tolist()])
img_pixel.shape
img_label=df2.breed
img_label=pd.get_dummies(df2.breed)
img_label.head()
breed_name = list(img_label.columns)

#%% load trained Mk1Mod model
from tensorflow import keras

Mk1model = keras.models.load_model('DBI-Keras/convnet_from_scratch.keras')
Mk1model.summary()

#%% load trained Mk2 model
from tensorflow import keras

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3))

Mk2model = keras.models.load_model('DBI-Keras-2/feature_extraction.keras')
Mk2model.summary()

#%% Mk1Mod predict
import cv2
import numpy as np

test_bgr1 = cv2.imread('Test/scot.jpg')
test_img1 = cv2.cvtColor(test_bgr1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(test_img1, (299, 299))

imgs = np.array([img1])
Mk1pred = Mk1model.predict(np.expand_dims(img1, axis=0))
print(breed_name[np.argmax(Mk1pred)])

#%% Mk2 predict
import cv2
import numpy as np

test_bgr1 = cv2.imread('Test/scot.jpg')
test_img1 = cv2.cvtColor(test_bgr1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(test_img1, (299, 299))

imgs = np.array([img1])

features = conv_base.predict(imgs)
Mk2pred = Mk2model.predict(features)
print(Mk2pred/np.linalg.norm(Mk2pred))
print(breed_name[np.argmax(Mk2pred)])

#%% plot
import matplotlib.pyplot as plt
plt.plot(np.arange(1,21),(np.squeeze(Mk1pred/np.linalg.norm(Mk1pred))),"bo",label="Mk1")
plt.plot(np.arange(1,21),(np.squeeze(Mk2pred/np.linalg.norm(Mk2pred))),"ro",label="Mk2")
plt.legend()
plt.show()

#%% imshow
test_bgr1 = cv2.imread('Test/scot.jpg')
test_img1 = cv2.cvtColor(test_bgr1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(test_img1, (299, 299))
plt.imshow(img1) 
ans = img_label.columns
