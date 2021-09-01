'''
Descripttion: 新的数据加载器;
                GameDataset同时完成检测与分割的数据处理;
                LoadImage;
                切图LoadImage;
version: 1.0
Author: SongJ
Date: 2021-07-31 08:59:16
LastEditors: SongJ
LastEditTime: 2021-09-01 08:55:13
'''

from torch.utils.data import Dataset
import os
from pathlib import Path
import cv2
import numpy as np
import glob
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

class GameDataset(Dataset):
    def __init__(self, 
                 img_path, 
                 img_size,
                 seg_imgsz=640,
                 stride=32):
        
        self.path = img_path
        self.img_size = img_size
        self.seg_imgsz = seg_imgsz
        
        # 加载数据文件
        # f=[]
        # p = Path(self.path)
        # with open(p,'r') as t:
        #     t = t.read().split()
        #     parent = str(p.parent) + os.sep
        #     f +=[x.replace('./', parent) if x.startswith('./') else x for x in t]
        path = img_path
        f = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic


            with open(p, 'r') as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                            # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
    
    
        self.image_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        
        
        ni = len(self.image_files) # number of images
        self.nf = ni  # number of files
        self.mode = 'images'
        
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * ni
        # 先不用
        

    def __iter__(self):
        self.count = 0
        return self
    
    def __getitem__(self, index):
        path = self.image_files[index]
        img0 = cv2.imread(path)
        img = letterbox(img0,new_shape=self.img_size)[0] 
        #print(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # 分割图像的处理
        seg_img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        seg_img = cv2.resize(seg_img,(self.seg_imgsz,self.seg_imgsz),interpolation=cv2.INTER_LINEAR)
        seg_img = seg_img / 255.0
        seg_img = seg_img.transpose((2, 0, 1))
        # if seg_img.ndimension() == 3:
        #     seg_img = seg_img.unsqueeze(0)
        seg_img = np.ascontiguousarray(seg_img, dtype = 'float32')

        return path, img, img0, seg_img
    
    def __len__(self):
        return self.nf
        
        
class LoadImages:  # YoloV5 for inference
    def __init__(self, path, img_size=640, stride=32):
        # f = []  # image files
        # p = path
        # p = Path(p)  # os-agnostic
        # with open(p, 'r') as t:
        #     t = t.read().strip().splitlines()
        #     parent = str(p.parent) + os.sep
        #     f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        #                 # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                # 加载数据文件
        f=[]
        p = Path(path)
        with open(p,'r') as t:
            t = t.read().split()
            parent = str(p.parent) + os.sep
            f +=[x.replace('./', parent) if x.startswith('./') else x for x in t]
        files=f
        
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        
        ni= len(images)
        self.img_size = img_size
        self.stride = stride
        self.files = images 
        self.nf = ni   # number of files
        self.mode = 'image'
        
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        return path, img, img0

    def __len__(self):
        return self.nf  # number of files

class CropLoadImages:  # YoloV5 Crop data for inference
    # warning ： 切图的返回值不同
    def __init__(self, path, img_size=640, stride=32):
        f = []  # image files

        p = Path(p)  # os-agnostic

        with open(p, 'r') as t:
            t = t.read().strip().splitlines()
            parent = str(p.parent) + os.sep
            f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        
        files=f
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni= len(images)

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = ni   # number of files
        self.video_flag = [False] * ni 
        self.mode = 'image'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]


        # Read image and crop
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        cH = int(img0.shape[0]/3) # 切掉的部分
        img1 = img0[cH:,:,:]     # 切图
        assert img0 is not None, 'Image Not Found ' + path
        #print(f'image {self.count}/{self.nf} {path}')

        # Padded resize
        #img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # 跟据切图的尺寸进行像素填充
        img = letterbox(img1, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img1, img0

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

