'''
Descripttion: only seg with plot;
            用于分割结果的可视化;
version: 1.0
Author: SongJ
Date: 2021-07-19 16:47:15
LastEditors: SongJ
LastEditTime: 2021-09-01 09:13:36
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
from pathlib import Path

# 检测模型环境
# import torch
# import torch.backends.cudnn as cudnn
# from models.experimental import attempt_load
# from utils.datasets import LoadImages
from utils.re_datasets import LoadImages
#from utils.reload import LoadImages
# from utils.general import check_file, check_img_size, check_requirements, \
#     box_iou, non_max_suppression, non_max_suppression1,scale_coords, xyxy2xywh, set_logging, colorstr
# from utils.torch_utils import select_device, time_synchronized
from utils.plots import colors, plot_one_box
# 分割模型环境
import paddle
import paddleseg.transforms as T
from paddle.inference import create_predictor
from paddleseg.cvlibs import manager
from paddle.inference import Config as PredictConfig
from paddleseg.utils.visualize import get_pseudo_color_map


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

    def run(self, imgs,path=None):

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        results = []


        data = imgs

        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results.append(output_handle.copy_to_cpu())

        return self.postprocess(results,path)

    def postprocess(self, results,path=None):


        results = np.concatenate(results, axis=0)

        
        for i in range(results.shape[0]):
            if opt.argmax:
                result_one = np.argmax(results[i], axis=0)
            else:                                                                                                                                                                                                                                       
                result_one = results[i]
            result = get_pseudo_color_map(result_one)
            basename = os.path.basename(path[0])
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.png'
            result.save(os.path.join(self.opt.save_dir, basename))
        
        return result_one

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0),3)
    return line_image

def hough(img):
    """
    args:img,seg results.
    return:hough line
    
    """
    # process
    img = cv2.cvtColor(img.astype('float32'),cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    
    # houg
    lines = cv2.HoughLinesP(img,1.85, np.pi/180,66, minLineLength=1,maxLineGap=250)
    
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2),(255,0,0),3)
    return line_image
    

def predict(opt):
    source,  view_img, save_txt, imgsz,seg_imgsz = opt.source, opt.view_img, opt.save_txt, opt.img_size,opt.seg_imgsz
    seg_stride=32 #segmodel的步长
    det_stride=32

    segmentor = Predictor(opt)
    
    
    # Load dataset
        # Load detect dataset
    dataset = LoadImages(source, img_size=imgsz, seg_imgsz = seg_imgsz, stride=det_stride)

        # Load seg dataset
    #img_files = get_img_list(source)


    # Run inference
        
    # if device.type != 'cpu':
    #     det_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(det_model.parameters())))  # run once

    # det time
    alltime,t0, t1 = 0., 0., 0.
    ts=time.time()
    result = {} # det ressults
    result["result"] = []
    #c_results = {"result": []} # seg results
    seg_dic = {1:8,2:9,3:10}
    start_time=time.time()
    # # det inference 
    for path, img, im0s,seg_img in dataset:
        start_time1 = time.time()
    #     # img0s是(h,w,c)
        
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0 3 letterbox img
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #     # Inference
    #     t = time_synchronized()
    #     pred = det_model(img, augment=opt.augment)[0]
    #     t0 += time_synchronized() - t
    #     # Apply NMS
    #     t = time_synchronized()
    #     pred = non_max_suppression1(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
    #                               max_det=opt.max_det)
    #     #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #     t1 += time_synchronized() - t
        
    #     # Process detections
    #     for i, det in enumerate(pred):
    #         p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
    #         image_id = p.split('/')[-1].split('.')[0]
    #         p = Path(p)  # to Path
    #         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    #         #imc = im0.copy() if opt.save_crop else im0 # for opt.save_crop
            
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
    #             # Write results
    #             for *xyxy, conf, cls in reversed(det):
    #                 result_one = {}
    #                 result_one["image_id"] = int(image_id)
    #                 if int(cls.item()) != 0 and int(cls.item()) != 1 and int(cls.item()) != 2 and int(cls.item()) != 9:
    #                     continue
    #                 result_one["type"] = int(cls.item()) + 1
    #                 result_one["x"] = xyxy[0].item() 
    #                 result_one["y"] = xyxy[1].item()
    #                 result_one["width"] = (xyxy[2].item() - xyxy[0].item())
    #                 result_one["height"] = (xyxy[3].item() - xyxy[1].item())
    #                 result_one["segmentation"] = []
    #                 result["result"].append(result_one)
                    
                
    # seg inference

        # 获取图片信息
        # im = cv2.imread(img)
        shape = im0s.shape[:2]

        image_id = int(path.split('/')[-1].split('.')[0])
        ##
        seg_img = seg_img[np.newaxis,:]
        
        # print("seg_img{}".format(seg_img.shape))
        # print("seg_imgtype{}".format(type(seg_img)))
        # print("dtype{}".format(seg_img.dtype))
        
        #seg_result = segmentor.run(im0s,[path])
        seg_result = segmentor.run(seg_img,[path])
        
        # 分割后处理
        line_img = hough(seg_result)

        #contours, hierarchy = cv2.findContours(seg_result.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        np.save('/home/aistudio/work/houghnpy/'+str(image_id)+'.npy',seg_result)
        
        #np.save('/home/aistudio/work/orgnpy/'+str(image_id)+'.npy',seg_result)
        w_ratio = float(shape[1]/seg_imgsz)
        h_ratio = float(shape[0]/seg_imgsz)
        
        # 原图绘画
        
        plotcontours = []
        for contour in contours:
            newcontour = contour.copy()
            for idx,point in enumerate(newcontour):
                newcontour[idx][0][0] = float(point[0][0] * w_ratio)
                newcontour[idx][0][1] = float(point[0][1] * h_ratio)
            plotcontours.append(newcontour)

        
        #print(shape[0])
        for idx,contour in enumerate(contours):
            segmentations = []
            rect = cv2.boundingRect(contour) # xywh
            
            for point in contour:
                segmentations.append(float(point[0][0] * w_ratio)) # x点的坐标
                segmentations.append(float(point[0][1] * h_ratio)) # y点的坐标
                #classes.append(seg_result[point[0][0],point[0][1]])

        
            
            # 计算类别
            seg_type =  Counter(seg_result[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]].flatten()).most_common(2)
            #print(seg_type)
            #判断类别
            if seg_type[0][0] == 0 and len(seg_type) > 1 :
                classes_type = seg_type[1][0]
            else:
                classes_type = seg_type[0][0]

            if classes_type!=1 and classes_type!=2 and classes_type!=3:
                continue

            #统计结果
            result_two = {}
            result_two["image_id"] = image_id
            result_two["type"] = int(seg_dic[classes_type])
            #result_two["type2"] = int(seg_dic[new_type])
            result_two["x"] = round(rect[0] * w_ratio, 2)
            result_two["y"] = round(rect[1] * h_ratio, 2)
            result_two["width"] = round(rect[2] * w_ratio, 2)
            result_two["height"] = round(rect[3] * h_ratio, 2)
            result_two["segmentation"] = [segmentations]
            result["result"].append(result_two)
            
        # 原图画出bbox----------------
            xyxy=[]
            xyxy.append(round(rect[0] * w_ratio, 1)) # x1
            xyxy.append(round(rect[1] * h_ratio, 1)) # y1
            xyxy.append(round(rect[0] * w_ratio, 1) + round(rect[2] * w_ratio, 1)) # 
            xyxy.append(round(rect[1] * h_ratio, 1) + round(rect[3] * h_ratio, 1))
            seg_colors = {8:(255,0,0),9:(0,255,0),10:(0,0,255)} 
            if view_img:  # Add bbox to image
                
                c = int(seg_dic[classes_type])  # integer class
                label=str(int(seg_dic[classes_type])) 
                #label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                plot_one_box(xyxy, im0s, label=label, color=colors(c, True), line_thickness=3)
                color = seg_colors[int(seg_dic[classes_type])]
                cv2.drawContours(im0s, plotcontours, idx,color,3) 
        
        p = Path(path)     
        if view_img:
            if not os.path.exists('results'):
                os.mkdir('results')
            save_path = os.path.join('results',p.name)
            cv2.imwrite(save_path, im0s)
        # 原图画出bbox------------------

        alltime =alltime+time.time()-start_time1
    print('per time:', alltime/40)

    print('Done:{}'.format(time.time()-start_time))
    if not os.path.exists(opt.output):
        os.mknod(opt.output)
    # 写文件
    with open(opt.output, 'w') as ft:
        json.dump(result, ft)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('source', type=str,default='data.txt',help='')
    parser.add_argument('output', type=str,default='result.json')
    #parser.add_argument('--seg_path', type=str,default='./model/fastscnn')
    #parser.add_argument('--seg_path', type=str,default='./PaddleSeg/output/bisennet480')
    #parser.add_argument('--seg_path', type=str,default='./PaddleSeg/output/bisennet480')
    #parser.add_argument('--seg_path', type=str,default='./model/deeplabv3p18')
    #parser.add_argument('--seg_path', type=str,default='./model/deeplabv3p34_640')
    #parser.add_argument('--seg_path', type=str,default='./model/deeplabv3p34_480')

    parser.add_argument('--seg_path', type=str,default='./model/deeplabv3p18_640')
    #parser.add_argument('--seg_path', type=str,default='./model/deeplabv3p18_769')
    #parser.add_argument('--seg_path', type=str, default='./PaddleSeg/fastscnnprune')
    #parser.add_argument('--seg_path', type=str, default='./PaddleSeg/fastscnnquant')
    #parser.add_argument('--seg_path', type=str, default='./PaddleSeg/testquant')
    #parser.add_argument('--weights', nargs='+', type=str, default='model/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=4, help='inference size (pixels)')
    parser.add_argument('--seg_imgsz', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--argmax', type=bool,default=False, help='')
    parser.add_argument('--save_dir', type=str,default='./outputpicture', help='')

    opt = parser.parse_args()
    print(opt)
    threshold = 0.05
    predict(opt=opt)
    
    
