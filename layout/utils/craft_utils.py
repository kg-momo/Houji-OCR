######################################
# CRAFTによる出力ファイルcraft/から、
#  imgxml/
#   img/
#   xml/
# 構成となるimgxmlファイルを作成する
######################################

import numpy as np
import cv2
from PIL import Image
import os
import glob

from . import get_lineArea as getLine
from . import get_segArea as getSeg
from . import make_imgxml as imgxml
#from . import rubi_hantei.hantei_histgram as hantei

def lineArea(reg, aff):

    # [0, 255] -> [0, 1] へ変換
    regionmap = (reg/255.).astype(np.float32)
    affmap = (aff/255.).astype(np.float32)
        
    link = 0.2 #aff
    low = 0.4 #reg
    text = 0.8
        
    boxes, polys, mapper, labels = getLine.getDetBoxes(regionmap, affmap, text_threshold=text, link_threshold=link, low_text=low, estimate_num_chars=False)

    return boxes, polys, mapper, labels

def segArea(reg, aff, segsize):
  
    # [0, 255] -> [0, 1] へ変換
    regionmap = (reg/255.).astype(np.float32)
    #regionmap = getSeq.getDilate(_regionmap, output_dir, name+"-reg", hor, ver)
    affmap = getSeg.getDilate(aff, segsize)
    
    link = 0.4 #aff
    low = 0.4 #reg
    text = 0.4
    
    boxes, polys, mapper, labels = getSeg.getDetBoxes(regionmap, affmap, text_threshold=text, link_threshold=link, low_text=low, estimate_num_chars=False)

    return boxes, polys, mapper, labels

def box_kakudai(hp, wp, box_p):
    
    bx, by = box_p[0][0], box_p[0][1]
    bw = box_p[1][0] - bx
    bh = box_p[3][1] - by
    
    
    bx, by = int(bx-wp), int(by-hp)
    if bx < 0: bx = 0
    if by < 0: by = 0
    bh = bh + hp + hp
    bw = bw + wp + wp

    kakudai_box_p = np.array([[bx,by],
                              [bx+bw, by],
                              [bx+bw, by+bh],
                              [bx, by+bh]])
    
    return kakudai_box_p
    
def imgxml_utils(input_img, reg_list, aff_list, out_dir, canvas_size):
    
    OUT_IMGXML = out_dir+"imgxml/"
    os.makedirs(OUT_IMGXML+"img/", exist_ok=True)
    os.makedirs(OUT_IMGXML+"xml/", exist_ok=True)

    
    for i, (reg, aff) in enumerate(zip(reg_list, aff_list)):
       
        l_boxes, _,_,_ = lineArea(reg, aff)
        
        img = input_img[i]
        #print("imgxml {:d}/{:d}: ".format(i+1, len(input_img), img), end="\r")
        imgxml.makeImageXML(img, canvas_size, OUT_IMGXML, l_boxes)
       
    print()

def seg_utils(name_list, reg_list, aff_list, out_dir, canvas_size, segsize):

    OUT_SEG = out_dir+"seg/"
    os.makedirs(OUT_SEG+"boxes/", exist_ok=True)
    os.makedirs(OUT_SEG+"xml_SegAndLine/", exist_ok=True)
    
    for i, (name, reg, aff) in enumerate(zip(name_list, reg_list, aff_list)):
        s_boxes, _,_,_ = segArea(reg, aff, segsize)
        #print(len(s_boxes))
        kakudai_boxes = []
        height, width = reg.shape[:2]
        hp = height//200
        wp = width//300
        for box_p in s_boxes:
            kakudai_boxes.append(box_kakudai(hp, wp, box_p))
        np.save(OUT_SEG+"boxes/"+name+"-np_boxes", np.array(kakudai_boxes))
        
        print("seg {:d}/{:d}: {:s}".format(i+1, len(name_list), name), end="\r")
        
        
    print()
