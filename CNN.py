#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:36:22 2020

@author: alikemalcelenk
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
#import warnings
import warnings
#filter warnings
warnings.filterwarnings('ignore')

#TOTAL 70.000 RESMIM VAR. 42000 TRAIN, 28000 TEST

#Read Train
train = pd.read_csv("train.csv")
print(train.shape) # (42000, 785) - (image, column)

#Read Test 
test= pd.read_csv("test.csv")
print(test.shape) #(28000, 784) | 785-1 olması sebebi labelin olmaması
#label = resimde yazan sayının tutulduğu kısım. Resimdeki sayı 3 se label 3 tür

#Y_train and X_train
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
#train datamdan, label dışında kalan bütün kısmınları alıp X_train diyorum

#Test Data Set
img = X_train.iloc[41200].to_numpy() 
# 41200. resmi alıp matrixe çevirdim
img = img.reshape((28,28))
# 28 e 28 lik bir matrix haline geldi
plt.imshow(img,cmap='gray')
plt.title(train.iloc[41200,0])
plt.axis("off")
plt.show()

#Normalization
X_train = X_train / 255.0
test = test / 255.0
#normalizationın tam formülü bu değildir fakat resimlerle çalıştığımızda 255 yeterli
print("x_train shape: ",X_train.shape) #(42000, 784)
print("test shape: ",test.shape)  #(42000, 784) 

#Reshape
#resimleri 28x28x1 formatına getirip 3D yapmak çünkü keras bu formatta çalışıyor 
#x1 = gray scale
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape) #(42000, 28, 28, 1)
print("test shape: ",test.shape) #(28000, 28, 28, 1)

#Label Encoding
#0 => [1,0,0,0,0,0,0,0,0,0]
#9 => [0,0,1,0,0,0,0,0,0,1]
from keras.utils.np_utils import to_categorical  #one-hot-encoding
Y_train = to_categorical(Y_train, num_classes = 10)

# Split Train and Validation
# Traini 2 ye ayırdım. %90 train %10 validation. Val ile testleri yapıcam
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2) 
print("x_train shape",X_train.shape) #(37800, 28, 28, 1)
print("x_test shape",X_val.shape) #(4200, 28, 28, 1)
print("y_train shape",Y_train.shape) #(37800, 10)
print("y_test shape",Y_val.shape) #(4200, 10)

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


#CNN MODEL
#conv => max pool => dropout =>     conv => max pool => dropout =>       fully connected (2 layer)

model = Sequential() #modeli oluşturduk.

#1
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
#filters = feature detectors
#kernel_size = filter length 
#keras için 28x28x1 matrixine çevirdik. 1 = gray scale
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu'))
#üstte verdiğimiz için artık burada input_shapei vermemize gerek yok
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#stride = 2 şer basamak atlıycaz
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
#Flatten -> matrixi düzleştirme işlemi. ANN için 
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax")) #output
#layerlar eklendi. activation functionlar relu ve softmax

#Adam Optimizer - change learning rate
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#Compiler
#loss functionı categorical_crossentropy ile buluyoruz. Eğer yanlış predict ederse loss yüksek, doğru predict ederse loss 0.
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

#Epoch and Batch Size
epochs = 10  
batchSize = 250

#Data Augmentation
datagen = ImageDataGenerator( #!!hyperparameters
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  #!randomly rotate images in the range 5 degrees
        zoom_range = 0.1, #!Randomly zoom image 10%
        width_shift_range=0.1,  #!randomly shift images horizontally 10%
        height_shift_range=0.1,  #!randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batchSize), epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batchSize)

# Evaluate the model
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()






