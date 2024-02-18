######################################
# ＜INPUT＞
#    一つ以上のjpg邦字新聞ファイル

# 1. img-preprocess
# 2. craft
# 3. craft-utils

#＜OUTPUT＞
#  craft/
#  imgxml/
#     img/
#     xml/
# 構成となるimgxmlファイルを作成する
######################################

import cv2
import numpy as np
import os
import glob
import sys
import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from img_preprocess import img_preprocess as pre
from craft import craft_test
from utils import craft_utils as utils

def get_options(args):
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                           help='Input Directory')
    parser.add_argument('--out_dir', type=str,
                           help='Output name')
    parser.add_argument('--canvas_size', type=int,
                           default=3000, help='CRAFT canvas size')
    parser.add_argument('--segsize', type=int,
                           default=30, help='segmentation box size')
    options = parser.parse_args(args)
    return options

def main():

    supported_img_ext = ["jpg", "png"]
    
    try:
        options = get_options(sys.argv[1:])
    except FileNotFoundError as err:
        print(err)
        sys.exit()
        
    INPUT_DIR = options.input_dir
    dirname = os.path.basename(os.path.dirname(INPUT_DIR))
     # check if input_root exists
    if not os.path.exists(INPUT_DIR):
        print('INPUT_ROOT not found :{0}'.format(INPUT_DIR), file=sys.stderr)
        exit(0)

    imgPath = []
    for ext in supported_img_ext:
        pt = [name for name in glob.glob(INPUT_DIR+"*."+ext)]
        imgPath.extend(pt)
    #print(imgPath)		
    if len(imgPath)==0:
        print('JPG or PNG FIlE not found :{0}'.format(INPUT_DIR), file=sys.stderr)
        exit(1)

    
    OUT_DIR = options.out_dir    
    os.makedirs(OUT_DIR, exist_ok=True)

    canvas_size = options.canvas_size
    segsize = options.segsize

    t = time.time()
    
    print("INPUT: ", INPUT_DIR)
    print("OUTPUT: ", OUT_DIR)
    print("canvas size: ", canvas_size)
    print("segmentation size: ", segsize)
    print("#############################################")
    
    
    ##### load_img #####
    pre_list = []
    filename_list = []
    for i, path in enumerate(imgPath):
        print("preprocess {:d}/{:d}: {:s}".format(i+1, len(imgPath), path), end='\r')
        # input image
        filename, _ = os.path.splitext(os.path.basename(path))
        filename_list.append(filename)
        img = cv2.imread(path)

        ##### img_preprocess #####
        elim = pre.elim_line(img)
        pre_img = cv2.bitwise_not(cv2.cvtColor(elim, cv2.COLOR_GRAY2RGB))
        pre_list.append(pre_img)

        
    ##### CRAFT #####  
    craft = craft_test.Test(pre_list, canvas_size, OUT_DIR, filename_list)
    reg_list, aff_list = craft.test_invg()
    
    #####   get LineArea   and   make IMGXML file   ##### 
    utils.imgxml_utils(imgPath, reg_list, aff_list, OUT_DIR, canvas_size)
    utils.seg_utils(filename_list, reg_list, aff_list, OUT_DIR, canvas_size, segsize)
    
    print("########### pred layout / file: {0} / time: {1} ###########".format(
        len(pre_list), time.time() - t))

if __name__ == '__main__':

    main()
    
    
