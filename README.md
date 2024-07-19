<!--
 * @Descripttion: your project
 * @version: 1.0
 * @Author: SongJ
 * @Date: 2021-09-01 09:22:38
 * @LastEditors: SongJ
 * @LastEditTime: 2021-09-01 09:56:37
-->
# [2021 IKCEST第三届“一带一路”国际大数据竞赛暨第七届百度&西安交大大数据竞赛三等奖(16名)](https://aistudio.baidu.com/aistudio/competition/detail/91)

## 比赛提交代码  
<font color=gray size=5></font>
分割模型使用PaddleSeg框架,在Aistudio完成训练  
检测模型使用基于YoloV5修改,在本地1080Ti/*4上完成训练  
分割模型推理代码由PaddleSeg/inference修改而来  

---
## 提交代码摘要  
### predict.py  

主要的提交代码,同时进行分割与检测的推理  

---
### predict_det.py and predict_seg.py  

单独提交检测模型的代码;单独提交分割模型的代码  
主要为了获取单个模型效果便于调参  

---
### predict_det_plot.py and predict_seg_plot.py  

分割与检测结果可视化的代码,用于调参
![image](https://github.com/SongJgit/2021-IKCEST/blob/main/results/00032.jpg)
![image2](https://github.com/SongJgit/2021-IKCEST/blob/main/results/08027.jpg)
