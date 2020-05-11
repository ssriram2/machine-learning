#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import keras
import numpy as np


# In[3]:


print(keras.__version__)


# In[4]:


print(np.__version__)


# In[7]:


get_ipython().run_cell_magic('python-v', '', '')


# In[2]:


def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


# In[3]:


def crop_generator(batches, crop_length):
    
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


# In[6]:


img_width, img_height = 256, 256
 
train_data_dir = '/home/ssriram2/5050/train'
validation_data_dir = '/home/ssriram2/5050/test'

epochs = 50
batch_size = 100
weight_decay = 0.004  # weight decay coefficient
CROP_LENGTH   = 227
 
if K.image_data_format() == 'channels_first':
    input_shape = (3, CROP_LENGTH, CROP_LENGTH)
else:
    input_shape = (CROP_LENGTH, CROP_LENGTH, 3)


# In[7]:


model = Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=input_shape))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output layer
model.add(Dense(2, kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(Activation('softmax'))


# In[8]:


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd,
              metrics =['accuracy'])


# In[9]:


train_datagen = ImageDataGenerator(
                rescale = 1. / 255,
                 shear_range = 0.2,
                  zoom_range = 0.2,
            horizontal_flip = True)
 
test_datagen = ImageDataGenerator(rescale = 1. / 255)
 
train_generator = train_datagen.flow_from_directory(train_data_dir,
                              target_size =(img_width, img_height),
                     batch_size = batch_size, class_mode ='categorical')
 
validation_generator = test_datagen.flow_from_directory(
                                    validation_data_dir,
                   target_size =(img_width, img_height),
          batch_size = batch_size, class_mode ='categorical')

train_crops = crop_generator(train_generator, CROP_LENGTH)
valid_crops = crop_generator(validation_generator, CROP_LENGTH)


# In[ ]:


model.fit_generator(train_crops,
    steps_per_epoch = train_generator.samples // batch_size,
    epochs = epochs, validation_data = valid_crops,
    validation_steps = validation_generator.samples//batch_size)


# In[10]:


model.summary()


# In[ ]:




