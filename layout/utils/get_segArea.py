# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import cv2
import math
from scipy.ndimage import label

import time

import os
import glob
import shutil


# Quad Boxの推定
# text_threshold=0.7
# low_text=0.4
# link_threshold=0.4
def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False):
    # prepare data
    linkmap = linkmap.copy() # linkmap -> Affinity Score map
    textmap = textmap.copy() # textmap -> Region Score map
    img_h, img_w = textmap.shape

    """ labeling method """
    # 閾値処理
    
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    # バイナリ画像取得
    #text_score_comb = np.clip(text_score + b255, 0, 1)
    text_score_comb = linkmap
    #text_score_comb = textmap

    # 連結領域、ラベル付け
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    ######ここまでがラベリング処理######
    
    det = []
    mapper = []
    for k in range(1,nLabels): # 連結領域を一つずつ処理
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        
        # 一定のサイズ以下であればnoizeとして削除
        if size < (img_w*img_h)//350: continue
        #if size < 20: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        if estimate_num_chars:
            _, character_locs = cv2.threshold((textmap - linkmap) * segmap /255., text_threshold, 1, 0)
            _, n_chars = label(character_locs)
            mapper.append(n_chars)
        else:
            mapper.append(k)
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        
        #### make box ####
        # セクション領域の推定
        # すべて長方形想定に変更している
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)

        # 変更：傾いていない外接する矩形領域

        _x, _y, _w, _h = cv2.boundingRect(np_contours)

        box = np.array([[_x, _y],
                        [_x+_w-1, _y],
                        [_x+_w-1, _y+_h-1],
                        [_x, _y+_h-1]]) # 4点の座標

        """
        # 外接矩形領域の取得
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        """
        
        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)

        # 追加: 細いの削除
        #if w < 100 or h < 200: continue
        # 追加: でかいの削除
        #if w > 800 or h > 2000: continue
        
        # ここでなにしてるかいまいちわからに
        # 幅が若干広がる あまり結果には影響しないかも
        if abs(1 - box_ratio) <= 5:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            
        # make clock-wise order
        # 座標は右回りで格納
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)
        
        det.append(box)

    return det, labels, mapper

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False, estimate_num_chars=False):
    if poly and estimate_num_chars:
        raise Exception("Estimating the number of characters not currently supported with poly.")
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper, labels

# 拡大処理
def getDilate(img, segsize):
        
    # バイナリデータに変換    
    #img_gray = cv2.bitwise_not(img)
    #bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -2)
    ret, bw = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    height, width = bw.shape[:2]

    # 横幅を計算
    #hor = width//70
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (segsize, height//1000))
    # dilateを実行
    dilate = cv2.dilate(bw, d_kernel)
    
    #cv2.imwrite("dilate.jpg", dilate)
    return dilate

"""    
if __name__ == "__main__":

    FILE_NAME = "../res_momo/elim_line/"

    
    INPUT_PATH = "elim_line/"

    OUT_NAME = "elim_line01/"
    output_dir = "./output/data_seg/"+OUT_NAME
    # サブディレクトリ該ファイルがあったら削除しておく
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    

    # 保存用の名前 filename(拡張子以外)と拡張子
    #filename, file_ext = os.path.splitext(os.path.basename(IMG_PATH))

    r_path = []
    a_path = []
    name_list = []
    for name in glob.glob(FILE_NAME+"*_aff_mask.jpg"):
        a_path.append(name)
        r_name = name.replace("_aff", "")
        r_path.append(r_name)
        _filename, _ = os.path.splitext(os.path.basename(r_name))
        filename = _filename.replace("_mask", "")
        name_list.append(filename)

        
    for i, name in enumerate(name_list):
        print(f"{i+1} / {len(name_list)}: {name}")

 
            
        _regionmap = np.array(Image.open(r_path[i]).convert('L'))
        _affmap = np.array(Image.open(a_path[i]).convert('L'))
        # [0, 255] -> [0, 1] へ変換
        regionmap = (_regionmap/255.).astype(np.float32)
        #affmap = (_affmap/255.).astype(np.float32)

        height, width = _affmap.shape[:2]
        print(height, width)
        
        hor = width//75
        ver = height//20
        
        regionmap = getDilate(_regionmap, output_dir, name+"-reg", hor, ver)

        affmap = getDilate(_affmap, output_dir, name+"-aff", hor, ver)
        #score = np.clip(affmap - regionmap, 0, 1)
        #cv2.imwrite(output_dir+name+"-score.jpg", score)
        
        start = time.perf_counter()
        
        link = 0.4 #aff
        low = 0.4 #reg
        text = 0.4
        
        boxes, polys, mapper, labels = getDetBoxes(regionmap, affmap, text_threshold=text, link_threshold=link, low_text=low, estimate_num_chars=False)
        
        np.save(output_dir+name+"-np_boxes", boxes)
        np.save(output_dir+name+"-np_polys", polys)
        np.save(output_dir+name+"-np_mapper", mapper)
        np.save(output_dir+name+"-np_labels", labels)
        
        print("time: ", time.perf_counter() - start)
"""
