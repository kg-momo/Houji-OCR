
"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
# EasyOCRのプログラム参照
# CRAFTの後処理プログラム
# 行切り出しするのに必要

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
import matplotlib.pyplot as plt

""" auxiliary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxiliary functions """


# Quad Boxの推定
# text_threshold=0.7
# low_text=0.4
# link_threshold=0.4
def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False):
    # prepare data
    linkmap = linkmap.copy() # linkmap -> Affinity Score map
    textmap = textmap.copy() # textmap -> Region Score map
    img_h, img_w = textmap.shape
    img_size = img_h*img_w
    """ labeling method """
    # 閾値処理
    newimg=(textmap.copy()*255).astype(np.uint8)
    #print(newimg.shape)
    blur = cv2.GaussianBlur(newimg,(1,1),0)  # Gauss
    maxpx = np.amax(blur)
    if maxpx !=0:
        maxscale = (blur * (255 / maxpx)).astype(np.uint8)
    else:
        maxscale=blur.copy()
    # 二値化処理を大津のしきい値選定手法に変更
    maxscale =cv2.medianBlur(maxscale, ksize=5) 
    if maxpx !=0:
        maxscale = (maxscale * (255 / maxpx)).astype(np.uint8)
    else:
        maxscale=maxscale.copy()
    ret, text_score = cv2.threshold(maxscale,0,255,cv2.THRESH_OTSU)
    #import pdb;pdb.set_trace()
    THRESHTXT=120
    b255=maxscale.copy()
    b255[b255>THRESHTXT]=255
    b255[b255<=THRESHTXT]=0
    
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    #ret, text_score = cv2.threshold(textmap.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    #ret, link_score = cv2.threshold(linkmap.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)

    # バイナリ画像取得
    text_score_comb = np.clip(b255 + link_score, 0, 1)
    #plt.imshow(text_score+link_score, cmap="gray")
    #plt.show()
    
    # 連結領域、ラベル付け
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    ######ここまでがラベリング処理######
    
    det = []
    mapper = []

    w_hist = [] ###
    
    for k in range(1,nLabels): # 連結領域を一つずつ処理
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        
        # 一定のサイズ以下であればnoizeとして削除
        #if size < 300: continue
        if size < img_size/5.3e+4: continue

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
        
        ###### make box ######
        # 単語領域の推定
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
        #if w < 20 or h < 10: continue
        if _w < img_w/165 or _h < img_h/500: continue
        # 追加: 横長のもの削除
        if _w > _h: continue
        # 追加: でかいの削除
        #if w > 800 or h > 2000: continue
        if _w > img_w/50 : continue

        # 斜め処理？
        if abs(1 - box_ratio) <= 0.1:
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

# Polygonの推定
def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment width is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False, estimate_num_chars=False):
    if poly and estimate_num_chars:
        raise Exception("Estimating the number of characters not currently supported with poly.")
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper, labels

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
"""
if __name__ == "__main__":

    FILE_NAME = "../res_momo/elim_line/"
    OUT_NAME = "elim_line/"
    output_dir = "./output/data/"+OUT_NAME
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
        affmap = (_affmap/255.).astype(np.float32)
        
        start = time.perf_counter()
        
        link = 0.4 #aff
        low = 0.4 #reg
        text = 0.8
        
        boxes, polys, mapper, labels = getDetBoxes(regionmap, affmap, text_threshold=text, link_threshold=link, low_text=low, estimate_num_chars=False)


        np.save(output_dir+name+"-np_boxes", boxes)
        np.save(output_dir+name+"-np_polys", polys)
        np.save(output_dir+name+"-np_mapper", mapper)
        np.save(output_dir+name+"-np_labels", labels)

        print("time: ", time.perf_counter() - start)
"""
