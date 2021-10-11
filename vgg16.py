# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:58:29 2021

@author: yu065_adadcw1
"""

import tensorflow as tf
from tensorflow.keras import layers,optimizers
import cv2
import os
import numpy as np

path="/home/bmllab/bml_pjh/project/for paper/data/new_train"
dir_=['/ext_range','/move_center','/ori']
last_folder=['/crop_1','/crop_2','/crop_3',"/crop_4"]
def read_data(path,dir_,last_folder):
    last_data=[]
    last_label=[]
    for num in range(len(dir_)):
        data=[]
        label=[]
        for i in range(len(last_folder)):
            os.chdir(path+dir_[num]+last_folder[i])
            file_names=os.listdir()
            for name in file_names:
                if name !=".ipynb_checkpoints":
                    img=cv2.imread(name,cv2.IMREAD_COLOR)
                    img=cv2.resize(img,(255,255))
                    data.append(img)
                    label.append(i)
        last_data.append(np.array(data))
        last_label.append(np.array(label))
        
    last_data=np.array(last_data)
    last_label=np.array(last_label)
    return last_data,last_label
#load_train_data
last_data,last_label=read_data(path,dir_,last_folder)

ext_train_data=last_data[0]
move_train_data=last_data[1]
ori_train_data=last_data[2]

ext_train_label=last_label[0]
move_train_label=last_label[1]
ori_train_label=last_label[2]

#load_test_data

path="/home/bmllab/bml_pjh/project/for paper/data/new_test"
last_data1,last_label1=read_data(path,dir_,last_folder)

ext_test_data=last_data1[0]
move_test_data=last_data1[1]
ori_test_data=last_data1[2]

ext_test_label=last_label1[0]
move_test_label=last_label1[1]
ori_test_label=last_label1[2]

#load_val_data
path="/home/bmllab/bml_pjh/project/for paper/data/new_val"
last_data2,last_label2=read_data(path,dir_,last_folder)

ext_val_data=last_data2[0]
move_val_data=last_data2[1]
ori_val_data=last_data2[2]

ext_val_label=last_label2[0]
move_val_label=last_label2[1]
ori_val_label=last_label2[2]


#vgg16 shape model
def vgg16(input_shape,classes):
    inp=tf.keras.layers.Input(shape=input_shape)
    
    x=tf.keras.layers.Conv2D(3,(3,3),activation="relu")(inp)
    
    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.Conv2D(64,(3,3),activation="relu")(x)
    x=tf.keras.layers.Conv2D(10,(1,1))(x)

    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.Conv2D(64,(3,3),activation="relu")(x)
    x=tf.keras.layers.Conv2D(10,(1,1))(x)

    x=tf.keras.layers.MaxPool2D((2,2),strides=2)(x)
    
    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.Conv2D(128,(3,3),activation="relu")(x)
    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.Conv2D(128,(3,3),activation="relu")(x)
    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.MaxPool2D((2,2),strides=2)(x)   
    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.Conv2D(512,(3,3),activation="relu")(x)
    x=tf.keras.layers.Conv2D(10,(1,1))(x)
    x=tf.keras.layers.Conv2D(512,(3,3),activation="relu")(x)
    x=tf.keras.layers.Conv2D(10,(1,1))(x)

    x=tf.keras.layers.MaxPool2D((2,2),strides=2)(x)
    
    x=layers.Flatten()(x)    
    x=layers.Dense(64,activation="relu")(x)
    x=layers.Dense(64,activation="relu")(x)
    pred=layers.Dense(classes,activation="softmax")(x) 
    model= tf.keras.models.Model(inputs=inp,outputs=pred)
    model.summary()
    return model


model_vgg16_ext=vgg16((255,255,3),4)
model_vgg16_move=vgg16((255,255,3),4)
model_vgg16_ori=vgg16((255,255,3),4)


optimizer=optimizers.Adam()
model_vgg16_ori.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

batch_size=32
epochs=15
model_vgg16_ori.fit(ori_train_data,ori_train_label,batch_size=batch_size,epochs=epochs,validation_data=(ori_val_data,ori_val_label),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
loss,acc=model_vgg16_ori.evaluate(ori_test_data,ori_test_label)
print("ori data test acc:%f"%acc)
os.chdir("\\home\\bmllab\\bml_pjh\\project\\for paper\\model\\pred_model")
model_vgg16_ori.save("vgg16_#1.h5")


model_vgg16_ext.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


model_vgg16_ext.fit(ext_train_data,ext_train_label,batch_size=batch_size,epochs=epochs,validation_data=(ext_val_data,ext_val_label),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
loss,acc=model_vgg16_ext.evaluate(ext_test_data,ext_test_label)
print("ext data test acc:%f"%acc)
os.chdir("\\home\\bmllab\\bml_pjh\\project\\for paper\\model\\pred_model")
model_vgg16_move.save("vgg16_#2.h5")

model_vgg16_move.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


model_vgg16_move.fit(move_train_data,move_train_label,batch_size=batch_size,epochs=epochs,validation_data=(move_val_data,move_val_label),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
loss,acc=model_vgg16_move.evaluate(move_test_data,move_test_label)
print("move data test acc:%f"%acc)
os.chdir("\\home\\bmllab\\bml_pjh\\project\\for paper\\model\\pred_model")
model_vgg16_move.save("vgg16_#3.h5")
