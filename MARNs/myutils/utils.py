import os
import numpy as np
from PIL import Image 
import imageio
from tqdm import tqdm


## 讀取data
def readfile(path) : 
    img_dir = sorted(os.listdir(path))  # path 為放 data 的 folder 
    x = np.ones((len(img_dir),512,512), dtype = np.uint8) # 512*512*3
    for i , file in enumerate(img_dir) :
        #print(file)
        img = Image.open(os.path.join(path,file))
        x[i] = img 
    return x

## 讀取gt 的data
def readgt(path) :
    img_dir = sorted(os.listdir(path))
    y  = np.ones((len(img_dir),512,512), dtype = np.uint8)
    y2 = np.ones((len(img_dir)*5,512,512), dtype = np.uint8)
    for i , file in enumerate(img_dir) :
        #print(file)
        img = Image.open(os.path.join(path,file))
        y[i] = img
    
    k = 0
    for i in range(0,len(img_dir)*5,5) :
        #print(k)
        y2[i] = y[k]
        y2[i+1] = y[k]
        y2[i+2] = y[k]
        y2[i+3] = y[k]
        y2[i+4] = y[k]
        k = k+1 
    
    return  y2



##自動建立目錄:
def auto_create_path(FilePath):
    if os.path.exists(FilePath):   ##目錄存在，不用建立
            print( '[Info] : dir exists'  ) 
    else:
            print( '[Info] : dir not exists') 
            os.makedirs(FilePath)  # 沒有目錄，就幫你建立

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def savefile(img,path_out,path_in) :
    img_dir = sorted(os.listdir(path_in))
    for i , file in enumerate(tqdm(img_dir)) :
        imageio.imwrite(os.path.join(path_out,'out_'+file),img[i])