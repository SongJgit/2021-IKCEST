'''
Descripttion: 仅提交分割模型;
version: 1.0
Author: SongJ
Date: 2021-07-19 16:47:15
LastEditors: SongJ
LastEditTime: 2021-09-01 09:00:53
'''

# 通用环境
import sys
sys.path.insert(0, '.')
import argparse
import time
from pathlib import Path
import yaml
import cv2
import os
import json
import codecs
import numpy as np
from collections import Counter

# 检测模型环境
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.reload import GameDataset
from utils.general import check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, non_max_suppression1,scale_coords, xyxy2xywh, set_logging, colorstr
from utils.torch_utils import select_device
from torch.utils.data import DataLoader
# 分割模型环境
import paddle
import paddleseg.transforms as T
from paddle.inference import create_predictor
from paddleseg.cvlibs import manager
from paddle.inference import Config as PredictConfig


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self._load_transforms(
            self.dic['Deploy']['transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)


class Predictor:
    def __init__(self,opt):
        self.cfg = DeployConfig(os.path.join(opt.seg_path, 'deploy.yaml'))
        self.opt = opt

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_glog_info()
        pred_cfg.enable_use_gpu(100, 0)
        self.predictor = create_predictor(pred_cfg)

    def preprocess(self, img):
        return self.cfg.transforms(img)[0]

    def run(self, imgs):

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        results = []

        #data = np.array([self.preprocess(imgs)])
        data= imgs
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results.append(output_handle.copy_to_cpu())

        return self.postprocess(results)
    
    def postprocess(self, results):

        # print('len(results){}'.format(len(results)))
        # print('results[0].shape(concat){}'.format(results[0].shape))
        results = np.concatenate(results, axis=0)
        # print('results.shape(concat){}'.format(results.shape))
        
        for i in range(results.shape[0]):
            if opt.argmax:
                result = np.argmax(results[i], axis=0)
            else:
                result = results[i]
        
        return result



def hough(img):
    """
    args:img,seg results.
    return:hough line
    
    """
    # process
    img = cv2.cvtColor(img.astype('float32'),cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    
    # hough
    lines = cv2.HoughLinesP(img, 1.8, np.pi/180, 90, minLineLength=1,maxLineGap=170)
    
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0),3)
    return line_image
    

def predict(opt):
    source, weights, view_img, imgsz,seg_imgsz = opt.source, opt.weights, opt.view_img, opt.img_size,opt.seg_imgsz
    seg_stride = 32 #segmodel的步长
    det_stride = 32

    # Initialize
    set_logging()
    device = select_device(opt.device) # 检测模型的device
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
        # Load seg_model
    segmentor = Predictor(opt)
        # Load detect_model
    #det_model = attempt_load(weights, map_location=device)
    #det_stride = int(det_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=det_stride)
    #names = det_model.module.names if hasattr(det_model, 'module') else det_model.names
    #if half:
    #  det_model.half()  # to FP16
    
    # Load dataset
    #dataset = LoadImages(source, img_size=imgsz, stride=det_stride)
    dataset = GameDataset(source, img_size=imgsz,seg_imgsz = seg_imgsz)
    dataloader = DataLoader(dataset,batch_size=1,num_workers=2)  # 尺寸过大时需要调整num_workers
    
    # Run inference
    #if device.type != 'cpu':
    #    det_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(det_model.parameters())))  # run once

    # det time
    t0, t1 = 0., 0.
    ts=time.time()
    result = {} # det ressults
    result["result"] = []
    seg_dic = {1:8,2:9,3:10}

    # inference 
    for path, img, im0s, seg_img in dataloader:
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).to(device)
        if torch.is_tensor(seg_img):
            seg_img =seg_img.numpy()
        if torch.is_tensor(im0s):
            im0s = im0s.squeeze(0).numpy()

        # img info
        image_id = int(path[0].split('/')[-1].split('.')[0])

        shape = im0s.shape[:2] # 用于seg还原图像,可能会出现是numpy的情况

        # img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0 3 letterbox img
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        
        # Inference
        # t = time.time()
        # pred = det_model(img, augment=opt.augment)[0]
        # t0 += time.time() - t
        # # Apply NMS
        # t = time.time()
        # pred = non_max_suppression1(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
        #                           max_det=opt.max_det)
        # #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t1 += time.time() - t
        
        # # Process detections
        # for i, det in enumerate(pred):
        #     p, s, im0, frame = path, '', im0s

        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.squeeze(0).numpy().shape).round()
    
        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
                    
        #             result_one = {}
        #             result_one["image_id"] = image_id
        #             result_one["type"] = int(cls.item()) + 1
        #             result_one["x"] = xyxy[0].item() 
        #             result_one["y"] = xyxy[1].item()
        #             result_one["width"] = (xyxy[2].item() - xyxy[0].item())
        #             result_one["height"] = (xyxy[3].item() - xyxy[1].item())
        #             result_one["segmentation"] = []
        #             result["result"].append(result_one)
                    
                
    # seg inference
        if len(seg_img.shape) ==2 :
            seg_img = seg_img[np.newaxis,:]


        seg_result = segmentor.run(seg_img)
        #seg_result = np.squeeze(seg_results[0, :, :]) * 80

        # 分割后处理
        line_img = hough(seg_result)

        # contours, hierarchy = cv2.findContours(seg_result.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w_ratio = float(shape[1]/seg_imgsz)
        h_ratio = float(shape[0]/seg_imgsz)
        for contour in contours:
            segmentations = []
            rect = cv2.boundingRect(contour)
            for point in contour:
                segmentations.append(round(point[0][0] * w_ratio, 1))
                segmentations.append(round(point[0][1] * h_ratio, 1))
            # 计算类别
            seg_type = Counter(seg_result[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]].flatten()).most_common(2)
            # 判断类别
            if seg_type[0][0] == 0 :
                classes_type = seg_type[1][0]
            else:
                classes_type = seg_type[0][0]

            # 判断是否是背景类
            if classes_type!=1 and classes_type!=2 and classes_type!=3:
                continue
            
            # 统计结果
            result_two = {}
            result_two["image_id"] = image_id
            result_two["type"] = int(seg_dic[classes_type])
            result_two["x"] = round(rect[0] * w_ratio, 1)  # 极有可能是填写宽高round(rect[0] * w_ratio, 1)
            result_two["y"] = round(rect[1] * h_ratio, 1)
            result_two["width"] = round(rect[2] * w_ratio, 1)
            result_two["height"] = round(rect[3] * h_ratio, 1)
            result_two["segmentation"] = [segmentations]
            result["result"].append(result_two)


    if not os.path.exists(opt.output):
        os.mknod(opt.output)
    # 写文件
    with open(opt.output, 'w') as ft:
        json.dump(result, ft)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('source',type=str,default='data.txt',help='')
    parser.add_argument('output',type=str,default='result.json')
    #parser.add_argument('--seg_path', type=str, default='model/fastscnn')
    #parser.add_argument('--seg_path', type=str, default='model/deeplabv3p18_512')
    parser.add_argument('--seg_path', type=str,default='./model/deeplabv3p34_640')
    parser.add_argument('--weights', nargs='+', type=str, default='model/best_80.pt', help='model.pt path(s)')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--seg_imgsz', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--argmax', type=bool,default=False, help='')  # 使用export的必须为false
    opt = parser.parse_args()
    print(opt)

    
    #start_time = time.time()
    
    
    
    predict(opt=opt)
    #print('total time:', time.time() - start_time)
