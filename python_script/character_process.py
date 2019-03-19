from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import os
import pandas as pd
import cv2
def imgtobinary(addr):
    img = Image.open(addr)
    #img = np.array(img.resize((400, 200)))
    img=np.array(img)
    characters = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    #binary = enhance(binary)
    return binary
def frame_cut(img):
    w,h=np.shape(img)
    left_top = [10000, 10000]
    right_bottom = [-1, -1]
    for i in range(w):
        for j in range(h):
            if img[i,j]==0:
                if left_top[0] > i:
                    left_top[0] = i
                if left_top[1] > j:
                    left_top[1] = j
                if right_bottom[0] < i:
                    right_bottom[0] = i
                if right_bottom[1] < j:
                    right_bottom[1] = j
    img = img[left_top[0]:right_bottom[0] + 1, left_top[1]:right_bottom[1] + 1]
    return img
def feature(img):
    n=4
    w,h=np.shape(img)
    rows=ceil(w/4)
    cols=ceil(h/4)
    matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            new_rows=min(w,(i+1)*rows)
            new_cols=min(h,(j+1)*cols)
            new_w=new_rows-i*rows
            new_h = new_cols - j*cols
            temp=img[i*rows:new_rows,j*cols:new_cols]
            matrix[i,j]=(new_w*new_h-np.sum(temp.reshape((1,-1))))/(new_w*new_h+0.1)
    array=matrix.reshape((1,-1))
    # #print(np.shape(array))
    # array_max=np.reshape(np.repeat(np.max(array,axis=1),n*n,axis=0),(1,-1))
    # array_min=np.reshape(np.repeat(np.min(array,axis=1),n*n,axis=0),(1,-1))
    # #print(np.shape(array_max))
    # array_range=array_max-array_min
    # array=(array-array_min)/array_range
    return array[0]

def generate_dataset(dir):
    name=os.listdir(dir)
    labels=[]
    datas=[]
    for temp in name:
        if temp[-3:]=='png':
            labels.append(ord(temp[0])-ord('a'))
            datas.append(feature(frame_cut(imgtobinary(dir+'/'+temp))))
            # print(temp)
    labels=(np.array(labels)).reshape((-1,1))
    datas=(np.array(datas)).reshape((-1,16))
    dataset=np.c_[datas,labels]
    csv_data = pd.DataFrame(dataset)
    csv_data.to_csv('csv_data1.csv',index=False)
    #print(dataset)
if __name__=='__main__':
    generate_dataset('lower1')
# dataset=[]
# labels=[]
# addrs=os.listdir()
# for addr in addrs:
#     if (os.path.splitext(addr))[-1]=='.png':
#         dataset.append(get_feture(addr))
#         if (os.path.splitext(addr))[0][5]=='1':

