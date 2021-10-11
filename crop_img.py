# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:38:44 2021

@author: yu065_adadcw1
"""

from PIL import Image
import os

def crop_img(dir,old_group,new_group):
    os.chdir(dir[:-1])
    for new_f_name in new_group:
        os.mkdir(new_f_name)
    old=[dir+old_group[i] for i in range(4)]
    after=[dir+new_group[i] for i in range(4)]
    for i in range(len(old)):
        os.chdir(old[i])
        img_files=os.listdir()
        for file_name in img_files:
            os.chdir(old[i])
            if file_name !=".ipynb_checkpoints":
                image1 = Image.open(file_name)
                croppedImage=image1.crop((55,40,385,250))
                os.chdir(after[i]) 
    
                croppedImage.save(file_name)
#crop original data
dir="/home/bmllab/bml_pjh/project/for paper/data/new_train/ori/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)

dir="/home/bmllab/bml_pjh/project/for paper/data/new_test/ori/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)

dir="/home/bmllab/bml_pjh/project/for paper/data/new_val/ori/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)


#crop move center data
dir="/home/bmllab/bml_pjh/project/for paper/data/new_train/move_center/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)

dir="/home/bmllab/bml_pjh/project/for paper/data/new_test/move_center/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)

dir="/home/bmllab/bml_pjh/project/for paper/data/new_val/move_center/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)


#crop ext_range data

dir="/home/bmllab/bml_pjh/project/for paper/data/new_train/ext_range/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)

dir="/home/bmllab/bml_pjh/project/for paper/data/new_test/ext_range/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)

dir="/home/bmllab/bml_pjh/project/for paper/data/new_val/ext_range/"
old_group=["1","2","3","4"]
new_group=['crop_1','crop_2','crop_3','crop_4']
crop_img(dir,old_group,new_group)
