# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:35:18 2021

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
    for i in range(num):
        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        new_data=one_circle(num,data_num,factor)
        new_data=make_larger(new_data,np.random.randint(100,150)/100)

        plus_x=int(4.5-np.max(new_data[:,0]))
        plus_y=int(4.5-np.max(new_data[:,1]))
        minus_x=np.min(new_data[:,0])
        minus_y=np.min(new_data[:,1])
        minus_x=int(-4.5-minus_x)
        minus_y=int(-4.5-minus_y)

        while(1):

            print('진입')
            plus_x_=np.random.randint(minus_x,plus_x)
            print(minus_x)
            print(minus_y)
            plus_y_=np.random.randint(minus_y,plus_y)
            if plus_x_!=0 or plus_y_!=0:
                break
        print(plus_x)
        print(plus_y)
                    
        print("plus_x:%d"%plus_x_)
        print("plus_y:%d"%plus_y_)
        print()    
            
        plt.scatter(new_data[:,0]+plus_x_,new_data[:,1]+plus_y_,c="k")
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("one_circle %d"%(i+10450))
        plt.cla()
            
        
        
def make_two1(num,path):
    print("two")
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
       
        plus_x=int(4.5-np.max(second_circle[:,0]))
        plus_y=int(4.5-np.max(second_circle[:,1]))
        minus_x=np.min(second_circle[:,0])
        minus_x=int(-4.5-minus_x)
        minus_y=np.min(second_circle[:,1])
        minus_y=int(-4.5-minus_y)
  
        iterable=True
        while(iterable):
            plus_x_=np.random.randint(minus_x,plus_x)
            plus_y_=np.random.randint(minus_y,plus_y)
            if plus_x_!=0 or plus_y!=0:
                break
            
        print("plus_x:%d"%plus_x)
        print("plus_y:%d"%plus_y)
        print()    
            
        plt.scatter(first_circle[:,0]+plus_x_,first_circle[:,1]+plus_y_,c="k")
        plt.scatter(second_circle[:,0]+plus_x_,second_circle[:,1]+plus_y_,c="k")        
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("two_circle %d"%(i+10501))
        plt.cla()



def make_three1(num,path):
    print("three")
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

        plus_x=int(4.5-np.max(third_circle[:,0]))
        plus_y=int(4.5-np.max(third_circle[:,1]))
        minus_x=np.min(third_circle[:,0])
        minus_x=int(-4.5-minus_x)
        minus_y=np.min(third_circle[:,1])
        minus_y=int(-4.5-minus_y)

        
        iterable=True
        while(iterable):
            plus_x_=np.random.randint(minus_x,plus_x)
            plus_y_=np.random.randint(minus_y,plus_y)
            if plus_x_!=0 or plus_y_!=0:
                break

        print("plus_x:%d"%plus_x)
        print("plus_y:%d"%plus_y)
        print()
        
        plt.scatter(first_circle[:,0]+plus_x_,first_circle[:,1]+plus_y_,c="k")
        plt.scatter(second_circle[:,0]+plus_x_,second_circle[:,1]+plus_y_,c="k")
        plt.scatter(third_circle[:,0]+plus_x_,third_circle[:,1]+plus_y_,c="k")

        
        plt.xlim(-4.5,4.5)
        plt.ylim(-4.5,4.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("three_circle %d"%(i+10230))
        plt.cla()        
        

        
def make_four1(num,path):
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
        
        plus_x=int(6.5-np.max(fourth_circle[:,0]))
        plus_y=int(6.5-np.max(fourth_circle[:,1]))
        minus_x=np.min(fourth_circle[:,0])
        minus_x=int(-6.5-minus_x)
        minus_y=np.min(fourth_circle[:,1])
        minus_y=int(-6.5-minus_y)

        iterable=True
        print(plus_x)
        print(minus_x)
        while(iterable):
            
            plus_x_=np.random.randint(minus_x,plus_x)
            plus_y_=np.random.randint(minus_y,plus_y)
            if plus_x_>=0.5 or plus_x_<=-0.5:
                if plus_y_>=0.5 or plus_y_<=-0.5:
                    break
        
        print("plus_x:%d"%plus_x_)
        print("plus_y:%d"%plus_y_)
        print()
    
        plt.scatter(first_circle[:,0]+plus_x_,first_circle[:,1]+plus_y_,c="k")
        plt.scatter(second_circle[:,0]+plus_x_,second_circle[:,1]+plus_y_,c="k")
        plt.scatter(third_circle[:,0]+plus_x_,third_circle[:,1]+plus_y_,c="k")
        plt.scatter(fourth_circle[:,0]+plus_x_,fourth_circle[:,1]+plus_y_,c="k")

        
        plt.xlim(-6.5,6.5)
        plt.ylim(-6.5,6.5)
        plt.ioff()
        os.chdir(path)
        plt.savefig("fourth_circle %d"%(i+10451))
        plt.cla()      
        
        
path="/home/bmllab/bml_pjh/project/for paper/data/new_train/move_center/"
make_one1(450,path+"1")
make_two1(450,path+"2")
make_three1(450,path+"3")
make_four1(450,path+"4")
        
path="/home/bmllab/bml_pjh/project/for paper/data/new_test/move_center/"
make_one1(90,path+"1")
make_two1(90,path+"2")
make_three1(90,path+"3")
make_four1(90,path+"4")

path="/home/bmllab/bml_pjh/project/for paper/data/new_val/move_center/"
make_one1(90,path+"1")
make_two1(90,path+"2")
make_three1(90,path+"3")
make_four1(90,path+"4")