# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:28:23 2021

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
        new_data=make_larger(new_data,np.random.randint(690,740)/100)

        last_data.append(new_data)
        plt.scatter(new_data[:,0],new_data[:,1],c="k")
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("one_circle %d"%(i))
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
        second_circle=make_larger(second_circle,np.random.randint(690,740)/100)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("three_circle %d"%(i))
        plt.cla()        
        




#small first circle,second circles
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
        second_circle=make_larger(second_circle,np.random.randint(220,270)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(690,740)/100)
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")

        
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("three_circle %d"%(i))
        plt.cla()        
        
        result.append(circle_data)

#small first circles
def make_three2(num,path):
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
        second_circle=make_larger(second_circle,np.random.randint(570,620)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(690,740)/100)
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")

        
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("three_circle %d"%(i+226))
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
        second_circle=make_larger(second_circle,np.random.randint(220,270)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(340,390)/100)
        
        factor=np.random.randint(7,8)
        factor=factor/10
    
        data_num=np.random.randint(30,40)
        fourth_circle=one_circle(num,data_num,factor)
        large_rate=np.random.randint(690,740)/100
    
        fourth_circle=make_larger(fourth_circle,large_rate)
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        circle_data=np.concatenate((circle_data,fourth_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")
        plt.scatter(fourth_circle[:,0],fourth_circle[:,1],c="k")

        
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("fourth_circle %d"%(i))
        plt.cla()        
        
        result.append(circle_data)

def make_four2(num,path):
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
        second_circle=make_larger(second_circle,np.random.randint(220,270)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(570,620)/100)
        
        factor=np.random.randint(7,8)
        factor=factor/10
    
        data_num=np.random.randint(30,40)
        fourth_circle=one_circle(num,data_num,factor)
        large_rate=np.random.randint(690,740)/100
    
        fourth_circle=make_larger(fourth_circle,large_rate)
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        circle_data=np.concatenate((circle_data,fourth_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")
        plt.scatter(fourth_circle[:,0],fourth_circle[:,1],c="k")

        
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("fourth_circle %d"%(i+151))
        plt.cla()        
        
        result.append(circle_data)
        
        
def make_four3(num,path):
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
        second_circle=make_larger(second_circle,np.random.randint(450,500)/100)

        factor=np.random.randint(7,8)
        factor=factor/10
        data_num=np.random.randint(30,40)
        third_circle=one_circle(num,data_num,factor)
        third_circle=make_larger(third_circle,np.random.randint(570,620)/100)
        
        factor=np.random.randint(7,8)
        factor=factor/10
    
        data_num=np.random.randint(30,40)
        fourth_circle=one_circle(num,data_num,factor)
        large_rate=np.random.randint(690,740)/100
        
        circle_data=np.concatenate((first_circle,second_circle),axis=0)
        circle_data=np.concatenate((circle_data,third_circle),axis=0)
        circle_data=np.concatenate((circle_data,fourth_circle),axis=0)
        
        plt.scatter(first_circle[:,0],first_circle[:,1],c="k")
        plt.scatter(second_circle[:,0],second_circle[:,1],c="k")
        plt.scatter(third_circle[:,0],third_circle[:,1],c="k")
        plt.scatter(fourth_circle[:,0],fourth_circle[:,1],c="k")

        
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.ioff()
        os.chdir(path)
        plt.savefig("fourth_circle %d"%(i+301))
        plt.cla()        
        
        result.append(circle_data)
        
        
        
dir="/home/bmllab/bml_pjh/project/for paper/data/new_train/ext_range/"
group=["1","2","3","4"]
make_one1(450,dir+group[0])
make_two1(450,dir+group[1])
make_three1(225,dir+group[2])
make_three2(225,dir+group[2])

make_four1(150,dir+group[3])
make_four2(150,dir+group[3])
make_four3(150,dir+group[3])

dir="/home/bmllab/bml_pjh/project/for paper/data/new_test/ext_range/"
make_one1(90,dir+group[0])
make_two1(90,dir+group[1])
make_three1(45,dir+group[2])
make_three2(45,dir+group[2])

make_four1(30,dir+group[3])
make_four2(30,dir+group[3])
make_four3(30,dir+group[3])

dir="/home/bmllab/bml_pjh/project/for paper/data/new_val/ext_range/"
make_one1(90,dir+group[0])
make_two1(90,dir+group[1])
make_three1(45,dir+group[2])
make_three2(45,dir+group[2])

make_four1(30,dir+group[3])
make_four2(30,dir+group[3])
make_four3(30,dir+group[3])
