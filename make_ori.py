# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:22:54 2021

@author: yu065_adadcw1
"""

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import os


def make_larger(circle,rate):
    new_circle=circle*rate
    return new_circle
def one_circle(num,data_num,factor):
    data,label=make_circles(n_samples=data_num,factor=factor, noise=0.03,shuffle=True)
    one_idx=[]
    for i in range(len(label)):
        if label[i]==0:
            one_idx.append(i)
    new_data=np.delete(data,one_idx,axis=0)
    return new_data

    
def make_one1(num,path):
    last_data=[]
    for i in range(num):
        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        new_data=one_circle(num,data_num,factor)
        new_data=make_larger(new_data,np.random.randint(100,150)/100)

        last_data.append(new_data)
        plt.scatter(new_data[:,0],new_data[:,1],c="k")
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("one_circle %d"%(i+450))
        plt.cla()
    

    
def make_two1(num,path):
    result=[]
    for i in range(num):
        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        first_circle=one_circle(num,data_num,factor)
        first_circle=make_larger(first_circle,np.random.randint(100,150)/100)


        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        second_circle=one_circle(num,data_num,factor)
        second_circle=make_larger(second_circle,np.random.randint(200,270)/100)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("three_circle %d"%(i+20000))
        plt.cla()        
        


def make_three1(num,path):
    result=[]
    for i in range(num):
        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        first_circle=one_circle(num,data_num,factor)
        first_circle=make_larger(first_circle,np.random.randint(100,150)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        second_circle=one_circle(num,data_num,factor)
        second_circle=make_larger(second_circle,np.random.randint(200,270)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(320,340)/100)
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")

        
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("three_circle %d"%(i+20000))
        plt.cla()        
        
        result.append(circle_data)


    
    
def make_four1(num,path):
    result=[]
    for i in range(num):
        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        first_circle=one_circle(num,data_num,factor)
        first_circle=make_larger(first_circle,np.random.randint(100,150)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        second_circle=one_circle(num,data_num,factor)
        second_circle=make_larger(second_circle,np.random.randint(200,270)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(320,340)/100)
        
        factor=np.random.randint(7,8)
        factor=factor/10
    
        data_num=np.random.randint(30,40)
        fourth_circle=one_circle(num,data_num,factor)
        large_rate=np.random.randint(390,440)/100
    
        fourth_circle=make_larger(fourth_circle,large_rate)
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        circle_data=np.concatenate((circle_data,fourth_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")
        plt.scatter(fourth_circle[:,0],fourth_circle[:,1],c="k")

        
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("fourth_circle %d"%(i+1000))
        plt.cla()        
        
        result.append(circle_data)
        
dir="/home/bmllab/bml_pjh/project/for paper/data/new_train/ori/"
group=["1","2","3","4"]
make_one1(450,dir+group[0])
make_two1(450,dir+group[1])
make_three1(450,dir+group[2])
make_four1(450,dir+group[3])

dir="/home/bmllab/bml_pjh/project/for paper/data/new_test/ori/"
make_one1(90,dir+group[0])
make_two1(90,dir+group[1])
make_three1(90,dir+group[2])
make_four1(90,dir+group[3])

dir="/home/bmllab/bml_pjh/project/for paper/data/new_val/ori/"
make_one1(90,dir+group[0])
make_two1(90,dir+group[1])
make_three1(90,dir+group[2])
make_four1(90,dir+group[3])