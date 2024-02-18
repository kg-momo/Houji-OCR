"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from . import craft_utilsg
from . import imgprocinv
from . import file_utils
import json
import zipfile
from PIL import Image, ImageFilter
from .craft import CRAFT

from collections import OrderedDict

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net, canvas_size, double_score_comb, mag_ratio, show_time):
    t0 = time.time()
    
    # resize
    img_resized, target_ratio, size_heatmap = imgprocinv.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgprocinv.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
    
        
    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    #####################################
        
    #print(score_text.shape)
    
    #heat=Image.fromarray(score_text,mode='RGB')
    #heat.save("heat.png")
    
    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()
    
    # Post-processing
    boxes, polys = craft_utilsg.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly, double_score_comb)
        
    # coordinate adjustment
    boxes = craft_utilsg.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utilsg.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    
    t1 = time.time() - t1
    ####################################only region score
    # render results (optional)
    render_img = score_text.copy()
    render_img2=score_link.copy()
    #render_img = np.hstack((render_img, score_link))
    ret_score_text = imgprocinv.cvt2HeatmapImg(render_img)
    ret_score_link = imgprocinv.cvt2HeatmapImg(render_img2)
    #Image.fromarray(imgproc.cvt2HeatmapImg(score_text)).save("heat.png")
    
    if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
        
    return boxes, polys, ret_score_text, ret_score_link

class Test:
    
    def __init__(self, img_list, canvas_size, resultf, filename_list):
        cd = os.getcwd()
        self.trained_model = cd+'/layout/craft/finetune_3dats_4b_10000epoch.pth'
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 1.0
        self.cuda = True
        self.canvas_size = canvas_size
        self.mag_ratio = 1.5
        self.poly = False
        #double_score_comb
        self.double_score_comb  = False
        self.show_time = False
        #self.test_folder = img_list
        self.image_list = img_list
        self.refine = False
        self.resultf = resultf+"/craft/"
        self.refiner_model ='./craft/weights/craft_refiner_CTW1500.pth'
        self.filename_list = filename_list

        """ For test images in a folder """
        #self.image_list, _, _ = file_utils.get_files(img_dir)
        self.result_folder = self.resultf #'./result/'
        if not os.path.isdir(self.result_folder):
            os.mkdir(self.result_folder)
            
   


    def test_invg(self):
        # load net
        net = CRAFT()     # initialize
    
        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            net.load_state_dict(copyStateDict(torch.load(self.trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(self.trained_model, map_location='cpu')))
        
        if self.cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        
        net.eval()
        
        # LinkRefiner
        refine_net = None
        if self.refine:
            from refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.refiner_model + ')')
            if self.cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model, map_location='cpu')))
            
            refine_net.eval()
            self.poly = True

        t = time.time()


        score_text_list = []
        score_link_list = []
        
        # load data
        for k, image in enumerate(self.image_list):
            print("CRAFT {:d}/{:d}: {:s}".format(k+1, len(self.image_list), self.filename_list[k]), end='\r')
            #image = imgprocinv.loadImage(image_path)
            
            bboxes, polys, score_text, score_link = test_net(
                net, image,
                self.text_threshold,
                self.link_threshold,
                self.low_text,
                self.cuda, self.poly,
                refine_net,
                self.canvas_size, self.double_score_comb,
                self.mag_ratio, self.show_time)


            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(self.filename_list[k]))
            mask_file = self.result_folder + "/res_" + filename + '_mask.jpg'
            mask_file2 = self.result_folder + "/res_" + filename + '_aff_mask.jpg'

            ###########################################
            cv2.imwrite(mask_file, cv2.resize(score_text , (int(score_text.shape[1]*2), int(score_text.shape[0]*2))))#score_text)
            cv2.imwrite(mask_file2, cv2.resize(score_link , (int(score_link.shape[1]*2), int(score_link.shape[0]*2))))#score_link)

            #file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=self.result_folder)

            score_text_list.append(cv2.resize(score_text , (int(score_text.shape[1]*2), int(score_text.shape[0]*2))))
            score_link_list.append(cv2.resize(score_link , (int(score_link.shape[1]*2), int(score_link.shape[0]*2))))
            
        print("elapsed time : {}s".format(time.time() - t))

        return score_text_list, score_link_list
