'''
Descripttion: 以paddleseg的工具为基础修改为适合2021ikcest比赛数据
version: 1.0
Author: PaddlePaddle Authors
Date: 2021-08-05 18:24:40
LastEditors: SongJ
LastEditTime: 2021-08-06 16:59:18
'''
# coding: utf8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

import argparse
import glob
import math
import json
import os
import os.path as osp
import numpy as np
import PIL.Image
import PIL.ImageDraw
import cv2

from gray2pseudo_color import get_color_map_list


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='input annotated directory')
    return parser.parse_args()


def main(args):
    output_dir = osp.join(args.input_dir, 'annotations')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating annotations directory:', output_dir)
    class_names = ['_background_','8','9','10'] # 比赛数据仅给了8,9,10.
    # 获取所有的类别，比赛的是8,9,10,分别代表实车道线，虚车道线，斑马线
    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        with open(label_file) as f:
            datas = json.load(f)
            for data in datas:
                if data['type'] == 8 or data['type'] == 9 or data['type'] == 10:
                        label = data['type']
                        cls_name = label
                        if not str(cls_name) in class_names:
                            class_names.append(cls_name)
    # 定义类别对应的索引bg:0,8:1,9:2,10:3
    class_name_to_id = {}
    for i, class_name in enumerate(class_names):
        class_id = i
        class_name_to_id[class_name] = class_id
        if class_id == 0:
            assert class_name == '_background_'
    class_names = tuple(class_names)
    print('class_names:', class_names)

    # 保存class_name.txt，保存的是8,9,10
    out_class_names_file = osp.join(args.input_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    # 获取颜色,画图
    color_map = get_color_map_list(256)
    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_png_file = osp.join(output_dir, base + '.png')

            data = json.load(f)
            print(base)
            img_file = osp.join(osp.dirname(label_file), base+'.jpg')
            img = np.asarray(cv2.imread(img_file))
            
            # 去掉不需要的标签
            data_save=[]
            for i,d in enumerate(data) : 
                if d['type'] ==8 or d['type'] ==9 or d['type'] ==10 :
                    data_save.append(d)
            
            # 对segmentation进行一下变形，符合labelme的格式
            for data in data_save:
                seg = data['segmentation']
                seg = redesign(seg)
                data['segmentation'] = seg


            lbl = shape2label(
                    img_size=img.shape,
                    shapes=data_save,
                    class_name_mapping=class_name_to_id
                )


            if osp.splitext(out_png_file)[1] != '.png':
                out_png_file += '.png'
            # Assume label ranges [0, 255] for uint8,
            if lbl.min() >= 0 and lbl.max() <= 255:
                lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')
                lbl_pil.putpalette(color_map)
                lbl_pil.save(out_png_file)
            else:
                raise ValueError(
                    '[%s] Cannot save the pixel-wise class label as PNG. '
                    'Please consider using the .npy format.' % out_png_file)

def shape2mask(img_size, points):
    label_mask = PIL.Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = PIL.ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, 'Polygon must have points more than 2'
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)


def shape2label(img_size, shapes, class_name_mapping):
    label = np.zeros(img_size[:2], dtype=np.int32)
    for shape in shapes:
        #points = shape['points']
        #print(points)
        points = shape['segmentation']
        #class_name = shape['label']
        class_name = str(shape['type'])
        shape_type = shape.get('shape_type', None)
        class_id = class_name_mapping[class_name]
        label_mask = shape2mask(img_size[:2], points)
        label[label_mask] = class_id
    return label

def redesign(data):

    seg_copy = data
    new_seg = []
    shape0 =len(seg_copy[0])
    for i in range(0,int(shape0),2):
        new_seg.append([seg_copy[0][i],seg_copy[0][i+1]])
    return new_seg

if __name__ == '__main__':
    args = parse_args()
    main(args)
