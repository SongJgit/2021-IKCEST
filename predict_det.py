'''
Descripttion: 仅提交检测模型;
version: 1.0
Author: SongJ
Date: 2021-07-18 09:42:55
LastEditors: SongJ
LastEditTime: 2021-09-01 09:00:36
'''

import sys
sys.path.insert(0, '.')
import argparse
import time
from pathlib import Path

import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.reload import GameDataset
from torch.utils.data import DataLoader
from models.experimental import attempt_load
from utils.general import check_img_size, \
    box_iou, non_max_suppression, non_max_suppression1,scale_coords, set_logging
from utils.torch_utils import select_device

import os
import json

def predict(opt):
    source, weights, view_img, imgsz,seg_imgsz = opt.source, opt.weights, opt.view_img, opt.img_size,opt.seg_imgsz
    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load detect_model
    det_model = attempt_load(weights, map_location=device)
    stride = int(det_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)
    names = det_model.module.names if hasattr(det_model, 'module') else det_model.names
    if half:
        det_model.half()  # to FP16

    # Load seg_model
    #segmentor = Segmentor(opt.seg_path)
    

    # Set Dataloader
    dataset = GameDataset(source, img_size=imgsz)
    dataloader = DataLoader(dataset,batch_size=1,num_workers=4,pin_memory=True)
    
    # Run inference
    if device.type != 'cpu':
        det_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(det_model.parameters())))  # run once
    
    ##
    #det_model.eval()
    t0, t1 = 0., 0.
    ts=time.time() # 总的起始时间
    result = {}
    result["result"] = []
    
    for path, img, im0s,seg_img in dataloader:
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).to(device)
        if torch.is_tensor(seg_img):
            seg_img =seg_img.numpy()
        if torch.is_tensor(im0s):
            im0s = im0s.squeeze(0).numpy()

        img = img.to(device)
        image_id = int(path[0].split('/')[-1].split('.')[0])
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 3 letterbox img
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
  
        print(im0s.shape)
        #Inference
        t = time.time()
        pred = det_model(img, augment=opt.augment)[0]
        t0 += time.time() - t  # 预测时间
        
        # Apply NMS
        t = time.time()
        pred = non_max_suppression1(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                  max_det=opt.max_det)
        #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t1 += time.time() - t #nms时间
        
        # Process detections
        for i, det in enumerate(pred):
            p, s, im0= path[0], '', im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],im0.shape).round()
    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    result_one = {}
                    result_one["image_id"] = image_id
                    result_one["type"] = int(cls.item()) + 1
                    result_one["x"] = xyxy[0].item() 
                    result_one["y"] = xyxy[1].item()
                    result_one["width"] = (xyxy[2].item() - xyxy[0].item())
                    result_one["height"] = (xyxy[3].item() - xyxy[1].item())
                    result_one["segmentation"] = []
                    result["result"].append(result_one)

    print(f'Done,all time. ({time.time() - ts:.3f}s)')
    print('Speed: %.1f/%.1f/%.1f s inference/NMS/total per %gx%g image' % (t0,t1,t0+t1,imgsz,imgsz))
    if not os.path.exists(opt.output):
        os.mknod(opt.output)
    with open(opt.output, 'w') as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('source',type=str,default='data.txt',help='')
    parser.add_argument('output',type=str,default='result.json')
    #parser.add_argument('seg_path',type=str,default='model/bisenet2/')
    parser.add_argument('--weights', nargs='+', type=str, default='model/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--seg_imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()
    print(opt)

    predict(opt=opt)
