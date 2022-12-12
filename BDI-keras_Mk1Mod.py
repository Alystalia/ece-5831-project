#!/usr/bin/env python
# coding: utf-8

# ECE-5831 Project
# Dog Breed Identifier -- Mk1 Modified
# Bowen Yu
# Comment: THe following work is inspired, learned and modified from https://www.kaggle.com/code/nafisur/dog-breed-identification-keras-cnn-basic
# Link to dataset: https://www.kaggle.com/c/dog-breed-identification


#%% 
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

df1=pd.read_csv('Kdata/labels.csv')
df1.head()
img_file='Kdata/train/'
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
X=img_pixel
y=img_label.values
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow(X_train,y=y_train,batch_size=32)
testing_set=test_datagen.flow(X_test,y=y_test,batch_size=32)

#%%
# Design and modify deep learning model
model=Sequential()

model.add(Conv2D(16, (3,3), input_shape=(299,299,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))  

model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2)) 

model.add(Conv2D(128, (3,3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))           
               
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.3))
model.add(Dense(20,activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()

#%%
# Start the training
from tensorflow import keras
callbacks = [keras.callbacks.ModelCheckpoint(filepath="convnet_from_scratch.keras",
                                            save_best_only=True,
                                            monitor="val_loss")]

history=model.fit(training_set,
                       batch_size = 32,
                       validation_data = testing_set,
                       epochs = 100,
                       verbose = 1,callbacks=callbacks)

#%%
# Plot the training history
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()