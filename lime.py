# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 23:47:09 2021

@author: yu065_adadcw1
"""

from lime import lime_image
import cv2
import numpy as np
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

def transform_img_fn2(train_x):
    out = []
    train_x
    for img_idx in range(train_x.shape[0]):
        img = train_x[img_idx]
        img = img/255.0
        x = np.expand_dims(img, axis=0)
        #x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


def make_lime_heatmap(explainer,model,segmenter,repeat_time,images,path):
    for i in range(repeat_time):
        explanation = explainer.explain_instance(images, 
                                             model.predict, 
                                             top_labels=5, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
        #Select the same class explained on the figures above.
        ind =  explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        if i==0:
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
            print(heatmap.shape)
            print(type(heatmap))
        else:
            heatmap2 = np.vectorize(dict_heatmap.get)(explanation.segments)
            heatmap+=heatmap2

        #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    os.chdir(path)
    plt.savefig("lime heatmap.png")
    
model=load_model("/home/bmllab/bml_pjh/project/for paper/model/pred_model/googlenet_#1.h5")
test_img_dir="/home/bmllab/bml_pjh/project/for paper/data/new_test/ext_range/"

explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('slic', kernel_size=1, max_dist=20000, ratio=0.2,n_segments=1000,)

image_list=os.listdir(test_img_dir)
value_list=[]
for img in image_list:
    value_list.append(cv2.imread(img,cv2.IMREAD_COLOR))
value_list=np.array(value_list)
images = transform_img_fn2(value_list)

make_lime_heatmap(explainer,model,segmenter,5,images[12],"/home/bmllab/bml_pjh/project/for paper/data/lime")