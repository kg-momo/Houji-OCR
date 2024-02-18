import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as gfg


# セクションごとにわけた出力と
# 行領域抽出結果を用いて
# セクションごとの文字情報保持データを作成する ndarray

# cross product: (b - a)×(c - a)
def cross3(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

# 図形内頂点包含判定
# 凸多角形の頂点は反時計回りのリストとして持つとする
# https://tjkendev.github.io/procon-library/python/geometry/point_inside_convex_polygon.html
def inside_convex_polygon(p0, qs):
    L = len(qs)
    left = 1; right = L
    q0 = qs[0]
    while left+1 < right:
        mid = (left + right) >> 1
        if cross3(q0, p0, qs[mid]) <= 0:
            left = mid
        else:
            right = mid
    if left == L-1:
        left -= 1
    qi = qs[left]; qj = qs[left+1]
    v0 = cross3(q0, qi, qj)
    v1 = cross3(q0, p0, qj)
    v2 = cross3(q0, qi, p0)
    if v0 < 0:
        v1 = -v1; v2 = -v2
        
    return (0 <= v1 and 0 <= v2 and v1 + v2 <= v0)

def BOX_inside_convex_polygon(box, qs):

    INSIDE = True
    for point in box:
        if inside_convex_polygon(point, qs): pass
        else:
            INSIDE = False
            return INSIDE

    return INSIDE

        
# xmlファイルからファイル名(img)を取り出す
def xml_getName(root, f_list, f_name):

    for pagename in root.iter("PAGE"):
        name = pagename.attrib["IMAGENAME"]
        f_list.append(name)
        _filename, _ = os.path.splitext(os.path.basename(name))
        f_name.append(_filename)

# xmlファイルからデータ(x,y,w,h,name)を取り出す
def xml_getData(root, datas, name):

    for filename in root:
        f_name = filename.attrib["IMAGENAME"]
        
        for line in filename.iter("LINE"):
            if (name == f_name):
                #print(name, f_name)
                s = line.attrib["STRING"]
                h = line.attrib["HEIGHT"]
                w = line.attrib["WIDTH"]
                x = line.attrib["X"]
                y = line.attrib["Y"]
                
                data = {"string": s, "height": int(h), "width": int(w),
                        "x": int(x), "y": int(y)}
                datas.append(data)

# line_xml_path-> output/$name/text/~.xml（共通）
# seg_box_path-> output/$name/seg/boxes/~.npy
# name-> imgxml imgfile name + jpg or png
def get_segdict(line_xml_path, seg_box_path, name):

    # 行文字認識結果
    # XMLfile
    xml_tree = gfg.parse(line_xml_path)
    # get root
    root = xml_tree.getroot()

    # 対象ファイルのxmlデータ取得
    lines = []
    xml_getData(root, lines, name)
    
    # セグメンテーション結果BOXのnpypath
    # 対象ファイルのsegboxデータ取得
    segboxes = np.load(seg_box_path)

    
    seg_dict = {}
    
    for j, segbox_p in enumerate(segboxes):
        #print("p:", segbox_p)
        inbox = []
        inbox.append(segbox_p)
        for line in lines:
            counter = 0
            s, h, w, x, y = line.values()
            points4 = np.array([[x, y],
                                [x+w-1, y],
                                [x+w-1, y+h-1],
                                [x, y+h-1]], dtype=np.float32)
            
            if BOX_inside_convex_polygon(points4, segbox_p):
                inbox.append(line)
        #print(len(inbox))
        seg_dict["box"+str(j)] = inbox
        
    return seg_dict


### seg_dict の構成 ###
"""

seg_dict: 辞書型
seg_dict{"box0": 0番目のBOX内包情報, ... ,"boxn"}

seg_dict[boxN][0] -> セクションボックスの座標情報が入ってる
seg_dict[boxN][1]~[num] -> boxNに内包されている行のname,x,y,w,hが辞書型で入ってる


"""
