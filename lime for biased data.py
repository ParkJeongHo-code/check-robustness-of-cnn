import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from skimage.segmentation import mark_boundaries
print('Notebook run using keras:', keras.__version__)

import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.models import load_model
import os
import cv2
print(os.getpid())

m1 = "googlenet_#1.h5"
m2 = "googlenet_#2.h5"
m3 = "googlenet_#3.h5"
os.chdir("/home/bmllab/bml_pjh/project/for paper/model/pred_model")
m1=load_model(m1)
m2=load_model(m2)
m3=load_model(m3)

%load_ext autoreload
%autoreload 2
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

os.chdir("/home/bmllab/bml_pjh/project/for paper/data/test_outlier/crop_3")
images = transform_img_fn([os.path.join('/home/bmllab/bml_pjh/project/for paper/data/new_train/ori_data/crop_2/three_circle 20005.png')])

test_img_pred.shape

for i in range(10230,10264):
    #bml_pjh/project/for paper/data/test_outlier/crop_1/one_circle 10453.png
    test_img =cv2.imread(f"/home/bmllab/bml_pjh/project/for paper/data/test_outlier/crop_3/three_circle {i}.png",cv2.IMREAD_COLOR)
    test_img =cv2.resize(test_img,(255,255))
    test_img_pred =test_img.reshape(1,test_img.shape[0],test_img.shape[1],3)
    result = m.predict(test_img_pred)
    print(result)

img_path = "/home/bmllab/bml_pjh/project/for paper/data/test_outlier/crop_4/"
img_file = "fourth_circle 10461.png"
#bml_pjh/project/for paper/data/test_outlier/crop_2/two_circle 10596.png
test_img =cv2.imread(img_path+img_file,cv2.IMREAD_COLOR)
test_img =cv2.resize(test_img,(255,255))
test_img_pred =test_img.reshape(1,test_img.shape[0],test_img.shape[1],3)
plt.imshow(test_img.astype('double'))

os.chdir("/home/bmllab/bml_pjh/project/for paper/xai")
directory = "/home/bmllab/bml_pjh/project/for paper/data/test_outlier/"

for path in ["crop_1/","crop_2/","crop_3/","crop_4/"]:
    img_path = directory + path
    count = 0
    for img_file in os.listdir(img_path):
        if count == 5:
            break
        count += 2
    #bml_pjh/project/for paper/data/test_outlier/crop_2/two_circle 10596.png
        test_img =cv2.imread(img_path+img_file,cv2.IMREAD_COLOR)
        test_img =cv2.resize(test_img,(255,255))
        test_img_pred =test_img.reshape(1,test_img.shape[0],test_img.shape[1],3)
        plt.imshow(test_img.astype('double'))

        result = []
        for i, model in enumerate([m1,m2,m3]):
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(test_img.astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
            result.append(explanation)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
            plt.imshow(mark_boundaries(temp / 2 + 0.1, mask, color=[0,0,55]))
            plt.imsave(f"./result_google/{img_file}_{i}.png", mark_boundaries(temp/255, mask,color=[0,0.33,0.77]))
            #plt.imsave(f"./result2/{img_file}_{i}.png", mark_boundaries(temp/255, mask,color=[0,0.33,0.77]))

        for i in range(3):
            explanation = result[i]
            #Select the same class explained on the figures above.
            ind =  explanation.top_labels[0]

            #Map each explanation weight to the corresponding superpixel
            dict_heatmap = dict(explanation.local_exp[ind])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
            
            #Plot. The visualization makes more sense if a symmetrical colorbar is used.
            #plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
            plt.imshow(heatmap, cmap = 'RdBu_r', vmin  = heatmap.min(), vmax = heatmap.max())
            plt.colorbar()
            plt.savefig(f'./result_google/heatmap/{img_file}_{i}_heatmap.png')
            #plt.savefig(f'./result2/heatmap/{img_file}_{i}_heatmap.png')
            plt.clf()    

#draw mean heatmap

os.chdir("/home/bmllab/bml_pjh/project/for paper/xai")
directory = "/home/bmllab/bml_pjh/project/for paper/data/test_outlier/"

for path in ["crop_1/","crop_2/","crop_3/","crop_4/"]:
    img_path = directory + path
    count = 0
    heatmap_save = []
    for img_file in os.listdir(img_path):
        if count == 2:
            break
        count += 1
    #bml_pjh/project/for paper/data/test_outlier/crop_2/two_circle 10596.png 
        test_img =cv2.imread(img_path+img_file,cv2.IMREAD_COLOR)
        test_img =cv2.resize(test_img,(255,255))
        test_img_pred =test_img.reshape(1,test_img.shape[0],test_img.shape[1],3)
        plt.imshow(test_img.astype('double'))

        result = []
        heatmaps = []
        for i, model in enumerate([m1,m2,m3]):
            heatmap_ = []
            explainer = lime_image.LimeImageExplainer()
            for j in range(5):
                print(f"adds heatmap_{j}", end=" ")
                explanation = explainer.explain_instance(test_img.astype('double')/255, model.predict, top_labels=5, hide_color=[0,0,0], num_samples=1000)
                ind =  explanation.top_labels[0]
                dict_heatmap = dict(explanation.local_exp[ind])
                heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
                heatmap_.append(heatmap)
            print(f"{model} adds heatmap_sum to heatmaps_{i}")
            heatmaps.append(sum(heatmap_))
            
        heatmap_save.append(heatmaps)
        for i in range(3):
            #Map each explanation weight to the corresponding superpixel
            #Plot. The visualization makes more sense if a symmetrical colorbar is used.
            #plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
            v = 0
            if(-sum(heatmaps[i]).min() < sum(heatmaps[i]).max()):
               v = sum(heatmaps[i]).max()
            else:
               v = -sum(heatmaps[i]).min()
            plt.imshow(heatmaps[i], cmap = 'RdBu_r', vmin  = -v, vmax = v)
            print(f"sum of heatmaps{i} is save process. heatmaps_{i} is sum of {len(heatmaps[i])}heatmaps.")
            plt.savefig(f'./result_google/heatmap_b2/{img_file}_{i}_heatmap.png')
            #plt.savefig(f'./result2/heatmap/{img_file}_{i}_heatmap.png')
            plt.clf()    

len(heatmap_save)

heatmaps= heatmap_save[0][2]
for i in range(3):
            #Map each explanation weight to the corresponding superpixel
            #Plot. The visualization makes more sense if a symmetrical colorbar is used.
            #plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
            v = 0
            if(sum(heatmaps[i]).min() < sum(heatmaps[i]).max()):
               v = sum(heatmaps[i]).max()
            else:
               v = -sum(heatmaps[i]).min()
            print(v)
            plt.imshow(heatmaps[i], cmap = 'RdBu_r', vmin  = -0.1, vmax = 0.1)
            print(f"sum of heatmaps{i} is save process. heatmaps_{i} is sum of {len(heatmaps[i])}heatmaps.")
            plt.savefig(f'./result_google/heatmap_b2/{img_file}_{i}_heatmap.png')
            #plt.savefig(f'./result2/heatmap/{img_file}_{i}_heatmap.png')
            plt.clf()    

heatmaps[2].min()

img_path = "/home/bmllab/bml_pjh/project/for paper/data/test_outlier/crop_1/"
img_file = "one_circle 10529.png"

test_img =cv2.imread(img_path+img_file,cv2.IMREAD_COLOR)
test_img =cv2.resize(test_img,(255,255))
test_img_pred =test_img.reshape(1,test_img.shape[0],test_img.shape[1],3)
plt.imshow(test_img.astype('double')/255)

plt.imshow(test_img.astype('double')/255)

temps = []
heatmaps = []

for i in range(5):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(test_img.astype('double')/255, m1.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    ind =  explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    temps.append(temp)
    heatmaps.append(heatmap)
    

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
#plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.imshow(sum(heatmaps), cmap = 'RdBu_r',
           vmin  = -sum(heatmaps).max(), vmax = sum(heatmaps).max())
plt.colorbar()
#plt.savefig(f'./result/heatmap/{img_file}_{i}_heatmap.png')

plt.imshow(heatmaps[4], cmap = 'RdBu_r', vmin  = -heatmaps[4].max(), vmax = heatmaps[4].max())
plt.colorbar()
plt.savefig(f'./result_google/heatmap_10/{img_file}_m3_heatmap.png')