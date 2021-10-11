# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:45:58 2021

@author: yu065_adadcw1
"""

import tensorflow as tf
from tensorflow.keras import optimizers
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

#resnet
def resnet(input_shape,classes):
    inputs=tf.keras.layers.Input(shape=input_shape)
    x=tf.keras.layers.Conv2D(32,(3,3),activation="relu")(inputs)

    x=tf.keras.layers.Conv2D(10,(1,1),activation="relu")(x)    

    x=tf.keras.layers.Conv2D(48,(3,3),activation="relu")(x)    
    
    x1=tf.keras.layers.Conv2D(14,(1,1),padding='same',activation='relu')(x)
    x2=tf.keras.layers.Conv2D(14,(3,3),padding='same',activation='relu')(x1)
    x3=tf.keras.layers.Conv2D(48,(1,1),padding='same',activation='relu')(x2)
    x4=tf.add(x3,x)   
    
    x11=tf.keras.layers.Conv2D(14,(1,1),padding='same',activation='relu')(x)
    x12=tf.keras.layers.Conv2D(14,(3,3),padding='same',activation='relu')(x11)
    x13=tf.keras.layers.Conv2D(48,(1,1),padding='same',activation='relu')(x12)
    x14=tf.add(x13,x)    
    x15=tf.keras.layers.Conv2D(14,(1,1),padding='same',activation='relu')(x14)
    x16=tf.keras.layers.Conv2D(14,(3,3),padding='same',activation='relu')(x15)
    x17=tf.keras.layers.Conv2D(48,(1,1),padding='same',activation='relu')(x16)
    x18=tf.add(x17,x14)

    x21=tf.keras.layers.Conv2D(14,(1,1),padding='same',activation='relu')(x)
    x22=tf.keras.layers.Conv2D(14,(3,3),padding='same',activation='relu')(x21)
    x23=tf.keras.layers.Conv2D(48,(1,1),padding='same',activation='relu')(x22)
    x24=tf.add(x23,x)
    x25=tf.keras.layers.Conv2D(14,(1,1),padding='same',activation='relu')(x24)
    x26=tf.keras.layers.Conv2D(14,(3,3),padding='same',activation='relu')(x25)
    x27=tf.keras.layers.Conv2D(48,(1,1),padding='same',activation='relu')(x26)
    x28=tf.add(x27,x24)
    x29=tf.keras.layers.Conv2D(14,(1,1),padding='same',activation='relu')(x28)
    x30=tf.keras.layers.Conv2D(14,(3,3),padding='same',activation='relu')(x29)
    x31=tf.keras.layers.Conv2D(48,(1,1),padding='same',activation='relu')(x30)
    x32=tf.add(x31,x28)    

    
    x=tf.add(x4,x18)
    x=tf.add(x,x32)

    
    x=tf.keras.layers.Conv2D(10,(1,1),activation="relu")(x)    
    x=tf.keras.layers.Conv2D(16,(3,3),activation="relu")(x)  
    x=tf.keras.layers.Conv2D(10,(1,1),activation="relu")(x)
    x=tf.keras.layers.Flatten()(x)    
    x=tf.keras.layers.Dense(128,activation="relu")(x)
    x=tf.keras.layers.Dense(64,activation="relu")(x)
    pred=tf.keras.layers.Dense(classes,activation="softmax")(x)
    
    model=tf.keras.models.Model(inputs=inputs,outputs=pred)
    print(model.summary())
    return model


model_resnet_ext=resnet((255,255,3),4)
model_resnet_move=resnet((255,255,3),4)
model_resnet_ori=resnet((255,255,3),4)


optimizer=optimizers.Adam()
model_resnet_ori.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

batch_size=32
epochs=15
model_resnet_ori.fit(ori_train_data,ori_train_label,batch_size=batch_size,epochs=epochs,validation_data=(ori_val_data,ori_val_label),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
loss,acc=model_resnet_ori.evaluate(ori_test_data,ori_test_label)
print("ori data test acc:%f"%acc)
os.chdir("\\home\\bmllab\\bml_pjh\\project\\for paper\\model\\pred_model")
model_resnet_ori.save("resnet_#1.h5")


model_resnet_ext.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


model_resnet_ext.fit(ext_train_data,ext_train_label,batch_size=batch_size,epochs=epochs,validation_data=(ext_val_data,ext_val_label),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
loss,acc=model_resnet_ext.evaluate(ext_test_data,ext_test_label)
print("ext data test acc:%f"%acc)
os.chdir("\\home\\bmllab\\bml_pjh\\project\\for paper\\model\\pred_model")
model_resnet_move.save("resnet_#2.h5")

model_resnet_move.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


model_resnet_move.fit(move_train_data,move_train_label,batch_size=batch_size,epochs=epochs,validation_data=(move_val_data,move_val_label),callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
loss,acc=model_resnet_move.evaluate(move_test_data,move_test_label)
print("move data test acc:%f"%acc)
os.chdir("\\home\\bmllab\\bml_pjh\\project\\for paper\\model\\pred_model")
model_resnet_move.save("resnet_#3.h5")
