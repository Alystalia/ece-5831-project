#!/usr/bin/env python
# coding: utf-8

# ECE-5831 Project
# Dog Breed Identifier -- Mk1
# Bowen Yu
# Comment: THe following work is inspired, learned and modified from https://www.kaggle.com/code/nafisur/dog-breed-identification-keras-cnn-basic

# ## Description
# 
# Link to dataset: https://www.kaggle.com/c/dog-breed-identification
# 
# The initiative model training progress is based on a small dataset.Kaggle’s dog breed identification challenge provides a decent dataset; itcontains over 20k images and comprise 120 breeds of dogs. These sets include training,testing, labels, and a submission sample file. In my implementation process, I onlyuse the training set and tags. After mapping the image id, breed(label), and filepath, I select the top 20 breeds based on quantities of images as the preliminarydataset. Then, I split the dataset into a train set and a test set at an 80/20ratio.

# ## Afterthoughts
# 
# The validation accuracy is increasing with training progress (at a prolonged rate), and achieving an acceptable value with a longer run time is possible. Meanwhile, I can also utilize the Stanford Dogs Dataset, another categorized/labeled dataset with a larger sample size. Since the degree of freedom in choosing models and modifying parameters under the Keras, it is worth trying a few and comparing the outcomes.

# ## Progress Demo

# ### Load libraries and dataset (images' id, breed and path)

# In[1]:


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

# In[2]:


df1=pd.read_csv('Kdata/labels.csv')
df1.head()


# In[3]:


img_file='Kdata/train/'


# In[4]:


df=df1.assign(img_path=lambda x: img_file + x['id'] +'.jpg')
df.head()


# ### Visualize the dataset

# In[15]:


df.breed.value_counts()
#ax=pd.value_counts(df['breed'],ascending=True).plot(kind='barh',fontsize="40",title="Class Distribution",figsize=(50,100))
#ax.set(xlabel="Images per class", ylabel="Classes")
#ax.xaxis.label.set_size(40)
#ax.yaxis.label.set_size(40)
#ax.title.set_size(60)
#plt.show()


# In[6]:


# Due to the time limitation, select top 20 breeds.
top_20=list(df.breed.value_counts()[0:20].index)
top_20 # Display top 20 breeds name.


# In[7]:


df2=df[df.breed.isin(top_20)]
df2.shape # Check total amount of images in the top_20 set.


# ### Convert images to numpy array of pixels

# In[8]:


img_pixel=np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in df2['img_path'].values.tolist()])
img_pixel.shape


# ### Label encoding dogs breed name (words to number)

# In[9]:


img_label=df2.breed
img_label=pd.get_dummies(df2.breed)
img_label.head()


# ### Train/Test set split

# In[10]:


X=img_pixel
y=img_label.values
print(X.shape)
print(y.shape)


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Pre-process the data in dataset for training

# In[12]:


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


# ### Design and modify deep learning model

# In[13]:


model=Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(299,299,3)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(20,activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()


# ### Start the training

# In[14]:
history=model.fit_generator(training_set,
                      steps_per_epoch = 32,
                      validation_data = testing_set,
                      validation_steps = 4,
                      epochs = 100,
                      verbose = 1)

#%%
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