import paddle.nn as nn
import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.io import Dataset,DataLoader
import glob
import os
from tqdm import tqdm
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class CitySegmentation(Dataset):
    def __init__(self):
        type_list=["jena","tubingen","monchengladbach","cologne","weimar","bochum","bremen","darmstadt","krefeld","hanover","dusseldorf","hamburg","strasbourg","erfurt","stuttgart","ulm","zurich","aachen"]
        self.data_list=[]
        self.gt_list=[]
        for i in type_list:
            data_path="cityscapes/leftImg8bit/train/"+i+"/"
            gt_path="cityscapes/gtFine/train/"+i+"/"
            self.data_list.extend(sorted(glob.glob(os.path.join(data_path, '*.png'))))
            self.gt_list.extend(sorted(glob.glob(os.path.join(gt_path, '*_labelTrainIds.png'))))


        self.total_size=len(self.data_list)
        self.palette = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[255]]

    def __getitem__(self,i):
        data=cv2.imread(self.data_list[i])
        data=data/255
        label=cv2.imread(self.gt_list[i],cv2.IMREAD_GRAYSCALE)
        label=label[:,:,np.newaxis]
        label=self.mask_to_onehot(label,self.palette)

        data=data.transpose((2,0,1))
        label=label.transpose((2,0,1))

        data_out=np.zeros((3,256,512),dtype="float32")
        label_out=np.zeros((20,256,512),dtype="float32")
        for j in range(3):
            data_out[j]=cv2.resize(data[j],(512,256))
        for j in range(20):
            label_out[j]=cv2.resize(label[j],(512,256))

        return data_out,label_out

    def __len__(self):
        return self.total_size

    def mask_to_onehot(self,mask, palette):
        semantic_map = []
        for colour in palette:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map