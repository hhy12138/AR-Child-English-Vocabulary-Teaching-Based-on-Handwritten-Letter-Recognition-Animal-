import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import math
def cohere(point,img):
    wp,hp=np.shape(point)
    wi,hi=np.shape(img)
    if hp<hi:
        complement=int((hi-hp)/2)
        right_complement=hi-hp-complement
        left_columns=np.ones((wp,complement))
        right_columns=np.ones((wp,right_complement))
        rows=np.ones((complement,hi))
        point=np.c_[left_columns,point]
        point=np.c_[point,right_columns]
        point=np.r_[point,rows]
        #point=np.r_[rows,point]
    else:
        complement = int((hp - hi) / 2)
        right_complement = hp - hi - complement
        left_columns = np.ones((wi, complement))
        right_columns = np.ones((wi, right_complement))
        img=np.c_[img,right_columns]
        img=np.c_[left_columns,img]
    return np.r_[point, img]
def average_space(imgs):
    total_space=0
    n=len(imgs)
    for img in imgs:
        total_space=space(img[0])
    return total_space/(n+0.01)
def space(img):
    w,h=np.shape(img)
    return w*h
def add_white_frame(img):
    w,h=np.shape(img)
    row=np.ones((1,h))
    column = np.ones((w+2,1))
    img=np.r_[img,row]
    img = np.r_[row,img]
    img=np.c_[column,img]
    img=np.c_[img,column]
    return img
def position(img):
    left=10000
    right=-1
    w,h=np.shape(img)
    for i in range(h):
        for j in range(w):
            if img[j,i]==0:
                left=min(left,i)
                right=max(right,i)
    return left,right
def smallest_window(img):
    w,h=np.shape(img)
    left_top = [10000, 10000]
    right_bottom = [-1, -1]
    for i in range(w):
        for j in range(h):
            if img[i,j]==0:
                if left_top[0]>i:
                    left_top[0]=i
                if left_top[1]>j:
                    left_top[1]=j
                if right_bottom[0]<i:
                    right_bottom[0]=i
                if right_bottom[1]<j:
                    right_bottom[1]=j
    img=img[left_top[0]:right_bottom[0]+1,left_top[1]:right_bottom[1]+1]
    return img
def enhance(img):
    w,h=np.shape(img)
    dirx=[-1,-1,-1,0,0,1,1,1]
    diry=dirx
    new_img=np.ones((w,h))
    for i in range(w):
        for j in range(h):
            if img[i,j]==0:
                new_img[i,j]=0
                for k in range(1):
                    new_x=i+dirx[k]
                    new_y = j + diry[k]
                    if new_x<0:
                        new_x=0
                    elif new_x>=w:
                        new_x=w-1
                    if new_y<0:
                        new_y=0
                    elif new_y>=h:
                        new_y=h-1
                    new_img[new_x,new_y]=0
    return new_img
def find_best_contrast(img):
    final_contrast=5
    min=10000000
    for contrast in [5,6,7,8,9]:
        temp_img = img.copy()
        enh_con = ImageEnhance.Contrast(temp_img)
        #contrast = contrast
        temp_img = enh_con.enhance(contrast)
        temp_img = np.array(temp_img.resize((400, 1000)))
        gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        #binary = enhance(binary)
        binary=add_white_frame(binary)
        binary = binary.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (np.shape(contours))[0]<min:
            min=(np.shape(contours))[0]
            final_contrast = contrast
    return final_contrast

def segment_character(addr):
    point_threshold=0.5
    img = Image.open(addr)
    #img.show()
    contrast=find_best_contrast(img)
    #print(contrast)
    # 对比度增强
    enh_con = ImageEnhance.Contrast(img)
    img = enh_con.enhance(contrast)
    #img.show()
    #img_contrasted.save("./0h/FGF2-new.tif")
    #img = cv2.imread(addr)
    img=np.array(img.resize((400,200)))

    characters=[]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray,5, 255, cv2.THRESH_BINARY)
    binary=enhance(binary)
    #plt.imshow(binary)
    #plt.show()
    #binary=smallest_window(binary)


    binary = add_white_frame(binary)
    binary = binary.astype(np.uint8)
    w, h = np.shape(binary)
    new_img = np.ones((w, h))
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_img = new_img.astype(np.uint8)
    for i in range(1,(np.shape(contours))[0]):
        cv2.drawContours(new_img, contours, i,0, cv2.FILLED)
    _, contours, hierarchy = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(1,(np.shape(contours))[0]):
        final = np.ones((w, h))
        cv2.drawContours(final, contours, i,0,cv2.FILLED)
        character=final+binary
        character[character>0]=1
        #mp is the middle position of a character
        left,right=position(character)
        middle=int((left+right)/2)
        character=smallest_window(character)
        if (np.shape(character))[0]*(np.shape(character))[1]<4:
            continue
        #print(right)
        characters.append([character,right])
    characters=sorted(characters,key=lambda k:k[1])
    average_sp=average_space(characters)
    #print(average_sp)
    i=0
    while i<len(characters):
        character=characters[i]
        #print(space(character[0]))
        if space(character[0])<point_threshold*average_sp:
            if i!=len(characters)-1:
                if abs(characters[i][1]-characters[i-1][1])>abs(characters[i+1][1]-characters[i][1]):
                    coh=cohere(character[0],characters[i+1][0])
                    characters[i]=[coh,characters[i+1][1]]
                    characters.pop(i+1)
                    i+=1
                else:
                    coh = cohere(character[0], characters[i - 1][0])
                    characters[i] = [coh, characters[i-1][1]]
                    characters.pop(i-1)
            else:
                coh = cohere(character[0], characters[i - 1][0])
                characters[i] = [coh, characters[i-1][1]]
                characters.pop(i - 1)
        i+=1
    # for character in characters:
    #     plt.imshow(character[0])
    #     plt.show()
    return characters

if __name__ == '__main__':
    segment_character('cat2.png')

