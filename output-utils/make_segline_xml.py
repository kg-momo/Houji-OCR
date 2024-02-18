# セクション情報を含んだxmlファイルを作成するプログラム

import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as gfg
import shutil


def AddElement(elem, width, height, x, y, s):
    #要素を追加
    line = gfg.SubElement(elem, "LINE",
                          {"CONF": "1.000", # よくわからん
                           "TYPE": "本文",
                           "HEIGHT": height,
                           "WIDTH": width,
                           "X": x,
                           "Y": y,
                           "STRING":s})

# 4点の座標から、xy-width-heightを算出するだけ
# 座標は右回りに格納されてるものとする
# 返り値は整数
def point4To_xywh(points):
    
    x, y = points[0][0], points[0][1]
    w = points[1][0] - x + 1
    h = points[3][1] - y + 1

    return int(x), int(y), int(w), int(h)


# 工事中   
def line_order(list_w, list_x, list_y):
    sort_index = np.argsort(np.array(list_x))[::-1]
    #print("sort:", sort_index)

    sameLine_list = [] # in same line index

    skip_index = []
    for i in range(len(list_x)):
        pnum = sort_index[i]
        if pnum in skip_index: continue
        pw = list_w[pnum]
        px = list_x[pnum]
        py = list_y[pnum]
        sameLine = [] # index
        sameLine_y = [] # y
        for k in range(i, len(list_x)):
            num = sort_index[k]
            w = list_w[num]
            x = list_x[num]
            y = list_y[num]
           
            if num == pnum:
                sameLine.append(num)
                sameLine_y.append(y)
            elif (x <= px <= x+w):
                sameLine.append(num)
                sameLine_y.append(y)
                skip_index.append(num)
            elif (px <= x+w <= px+pw):
                sameLine.append(num)
                sameLine_y.append(y)
                skip_index.append(num)

        if len(sameLine_y) > 1:
            sort_sameLY = np.argsort(np.array(sameLine_y))
            sort_sameL = []
            for sorty in sort_sameLY:
                sort_sameL.append(sameLine[sorty])
            sameLine_list.append(sort_sameL)
            continue
            
        sameLine_list.append(sameLine)
    new_sort_index = sum(sameLine_list, [])
    #print("new:", new_sort_index)
    return new_sort_index
        
"""
    sort_index = np.argsort(np.array(list_x))[::-1]
    new_sort_index = []

    for pnum in sort_index:
        pw = list_w[pnum]
        px = list_x[pnum]
        py = list_y[pnum]
        #for i, (w, x, y) in enumerate(zip(list_w, list_x, list_y)):
        for i, num in enumerate(sort_index):
            w = list_w[num]
            x = list_x[num]
            y = list_y[num]
            fy_index_list = []
            if num == pnum: continue
            if (x <= px <= x+w) and (py > y) :
                fy_index_list.append(i)
                continue
            if (px <= x+w <= px+pw) and (py > y):
                fy_index_list.append(i)
                continue
            
        if len(fy_index_list)==0: continue
"""         


#file_path->output/$name/seg/
def segline_xml(resize_path, seg_dict, file_path):
    
    # リサイズ済み解析対象の画像データfailpass
    # 画像読み込み
    preimg = cv2.imread(resize_path, cv2.IMREAD_GRAYSCALE)
    # name
    pred_n = os.path.basename(resize_path)    
    # height, width
    pred_h, pred_w = preimg.shape

    #xmlファイル保存name
    XML = file_path+"xml_SegAndLine/"+pred_n+".xml"
    
    root = gfg.Element("OCRDATASET")
    page = gfg.SubElement(root, "PAGE",
                          {"HEIGHT": str(pred_h),
                           "IMAGENAME": pred_n,
                           "WIDTH": str(pred_w)})
    
    ######################################################
    # segdictデータの読み込み
    for j, box_name in enumerate(seg_dict):
        points =  seg_dict[box_name][0]
        bx, by, bw, bh = point4To_xywh(points)

        segBox = gfg.SubElement(root, "BOX",
                                {"BOXNAME": box_name,
                                 "X": str(bx),
                                 "Y": str(by),
                                 "WIDTH": str(bw),
                                 "HEIGHT": str(bh)})

        line_list = []
        list_w = []
        list_x = []
        list_y = []
        for k in range(1, len(seg_dict[box_name])-1):
            line = seg_dict[box_name][k]
            s, h, w, x, y = line.values()
            #print(s, h)
            line_list.append([s, h, w, x, y])
            list_w.append(w)
            list_x.append(x)
            list_y.append(y)
            
        line_read_index = line_order(list_w, list_x, list_y)
        #line_read_index = np.argsort(np.array(list_x))
        for k in line_read_index:
            s, h, w, x, y = line_list[k]
            AddElement(segBox, str(w), str(h), str(x), str(y), s) # 要素追加
##                 
    tree = gfg.ElementTree(root)
        
    with open (XML, "wb") as files :
        tree.write(files)

    return XML

#if __name__ == "__main__":
#    main()
