'''
Descripttion: det crop data with plot;
                用于切图检测的推理代码;
                发现效果一般后废弃;
version: 1.0
Author: SongJ
Date: 2021-07-18 09:42:55
LastEditors: SongJ
LastEditTime: 2021-09-01 08:57:41
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

from models.experimental import attempt_load
from utils.re_datasets import LoadImages
from utils.general import check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, non_max_suppression1,scale_coords, xyxy2xywh, set_logging, colorstr
from utils.torch_utils import select_device, time_synchronized
from utils.plots import colors, plot_one_box
import os
import json







def predict(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
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
    
    
    
    
    
    vid_path, vid_writer = None, None
    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    # Run inference
    if device.type != 'cpu':
        det_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(det_model.parameters())))  # run once
    
    
    ##
    t0, t1 = 0., 0.
    
    result = {}
    result["result"] = []
    
    for path, img, im0s, vid_cap ,oimg in dataset:
        cH = int(oimg.shape[0]/3)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0 3 letterbox img
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t = time_synchronized()
        pred = det_model(img, augment=opt.augment)[0]
        t0 += time_synchronized() - t
        # Apply NMS
        t = time_synchronized()
        pred = non_max_suppression1(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                  max_det=opt.max_det)
        #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t1 += time_synchronized() - t
        
        # Process detections
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            image_id = p.split('/')[-1].split('.')[0]
            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            #imc = im0.copy() if opt.save_crop else im0 # for opt.save_crop
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy[1]=xyxy[1]+cH
                    xyxy[3]=xyxy[3]+cH
                    result_one = {}
                    result_one["image_id"] = int(image_id)

                    result_one["type"] = int(cls.item()) + 1
                    result_one["x"] = xyxy[0].item() 
                    result_one["y"] = xyxy[1].item()
                    result_one["width"] = (xyxy[2].item() - xyxy[0].item())
                    result_one["height"] = (xyxy[3].item() - xyxy[1].item())
                    result_one["segmentation"] = []
                    result["result"].append(result_one)

                    # if view_img:  # Add bbox to image
                    #     #save_dir=
                    #     c = int(cls.item()) + 1  # integer class
                    #     label=str(c+1)
                    #     #label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    #     plot_one_box(xyxy, oimg, label=label, color=colors(c, True), line_thickness=3)
                      
            # if view_img:
            #     if not os.path.exists('results'):
            #         os.mkdir('results')
            #     save_path = os.path.join('results',p.name)
            #     cv2.imwrite(save_path, oimg)
    ###seg_model
    
    
    
    
    
    
    
    
    print(f'Done. ({time.time() - ts:.3f}s)')
    
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image' % (t0,t1,t0+t1,imgsz,imgsz))
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
    parser.add_argument('--img_size', type=int, default=800, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()
    print(opt)
    
    ts=time.time()
    predict(opt=opt)
    print(f'Done. ({time.time() - ts:.3f}s)')