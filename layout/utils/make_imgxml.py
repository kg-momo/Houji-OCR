# NDLOCRに対応したxmlファイルを作成するためのプログラム
# ルビ除去関数も追加

import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as gfg
#import rubi
import shutil
import matplotlib.pyplot as plt
from .fourier_rubi_hantei import hantei_histgram as hantei
#import rubi_hantei.img_preprocess as pre
#import rubi_hantei.edge_fourier as edg


def AddElement(elem, width, height, x, y):
    #要素を追加
    line = gfg.SubElement(elem, "LINE",
                          {"CONF": "1.000", # よくわからん
                           "HEIGHT": height,
                           "TYPE": "本文",
                           "WIDTH": width,
                           "X": x,
                           "Y": y})

# CRAFTで使われてるリサイズ関数
def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    
    height, width, channel = img.shape
    
    # magnify image size
    target_size = mag_ratio * max(height, width)
    
    # set original image size
    if target_size > square_size:
        target_size = square_size
    
    ratio = target_size / max(height, width)    

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)


    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized

def makeImageXML(img_path, canvas_size, file_path, boxes):

    name, _ =  os.path.splitext(os.path.basename(img_path))

    # imgファイルへの保存
    img = cv2.imread(img_path)
    resize_img = resize_aspect_ratio(img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)

    cv2.imwrite(file_path+"img/"+name+".jpg", resize_img)

    #xmlファイルへの保存
    XML = file_path+"xml/"+name+".xml"
    height, width = resize_img.shape[:2]
    
    root = gfg.Element("OCRDATASET")
    page = gfg.SubElement(root, "PAGE",
                          {"HEIGHT": str(height),
                           "IMAGENAME": name+".jpg",
                           "WIDTH": str(width)})
    # 行領域の取得
    for j in range(len(boxes)):
        print("{:s}: line {:d}/{:d}".format(name, j+1, len(boxes)), end="\r")
        plot = boxes[j].astype('uint64')
        _x, _y = plot[0][0], plot[0][1]
        _h = int(plot[3][1] - _y + 1)
        _w = int(plot[1][0] - _x + 1)

        # 行領域
        trim_img = resize_img[_y:int(_y+_h+1), _x:int(_x+_w+1), :]

        # ルビ除去
        if ((width//100 <= _w <= width//35) and (height//100 <= _h <= height//7)):
            split = hantei.get_rubi_hantei(trim_img)            
            if not(split): # if False
                #cv2.imwrite("./output/trim/"+name+"/nashi/"+str(j)+".jpg", trim_img)
                pass
            else:
                _w = int(split)
                new_img = trim_img[: ,:_w]
                # ルビ除去後画像保存 必要なければ消すこと
                #cv2.imwrite("./output/trim/"+name+"/ari/"+str(j)+".jpg", new_img)
        # 左方向，上方向，下方向に拡大する
        hp = height//500
        wp = width//330
        x, y = int(_x-wp), int(_y-hp)
        if x < 0: x = 0
        if y < 0: y = 0
        h = _h + hp + hp
        w = _w + wp
        
        AddElement(page, str(w), str(h), str(x), str(y)) # 要素追加
            
        tree = gfg.ElementTree(root)
        
        with open (XML, "wb") as files :
            tree.write(files) 

"""
if __name__ == "__main__":

    INPUT_PATH = "elim_line/"
    DATA_PATH = "./output/data/"+INPUT_PATH # boxesのデータ入ったやつ
    OUT_F = "elim_line_rubi/"

    box_path = []
    name_list = []


    output_dir = "./output/imgxml/"+OUT_F
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"img/", exist_ok=True)
    os.makedirs(output_dir+"xml/", exist_ok=True)

    rubi_output_dir = "./output/imgxml/trim_rubi/"+OUT_F
    if os.path.exists(rubi_output_dir):
        shutil.rmtree(rubi_output_dir)
    os.makedirs(rubi_output_dir, exist_ok=True)
    
    hist_output_dir = "./output/imgxml/hist/"+OUT_F
    if os.path.exists(hist_output_dir):
        shutil.rmtree(hist_output_dir)
    os.makedirs(hist_output_dir, exist_ok=True)
    
    for name in glob.glob(DATA_PATH+"*-np_boxes.npy"):
        box_path.append(name)
        _filename, _ = os.path.splitext(os.path.basename(name))
        print(_filename)
        filename = _filename.replace("-np_boxes", "")
        name_list.append(filename)

    for i, name in enumerate(name_list):
        print(f"{i} / {len(name_list)}: {name}")
        # リサイズ済み解析対象の画像データfailpass
        pred_path = glob.glob("./output/resize/"+name+"*")[0]
        # 画像読み込み
        preimg = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # name
        pred_n = os.path.basename(pred_path)
        # height, width
        pred_h, pred_w = preimg.shape
        # black white
        ret, bw_img = cv2.threshold(preimg, 0, 255, cv2.THRESH_OTSU)

        # imgファイルへの保存
        cv2.imwrite(output_dir+"img/"+pred_n, preimg)       

        root = gfg.Element("OCRDATASET")
        page = gfg.SubElement(root, "PAGE",
                              {"HEIGHT": str(pred_h),
                               "IMAGENAME": pred_n,
                               "WIDTH": str(pred_w)})
        
        
        # QuadBoxのデータ読み込み
        # このデータを利用してXMLファイルを作成する
        # boxesの各要素(plot)-> [0]:左上座標 [1]:右上 [2]:右下 [3]:左下
        # 左上始点なので、座標(x,y)は plot[0] -> [0]:x [1]:y
        boxes = np.load(box_path[i])

       

        for j in range(len(boxes)):
            plot = boxes[j].astype('uint64')
            _x, _y = plot[0][0], plot[0][1]
            _h = int(plot[3][1] - _y + 1)
            _w = int(plot[1][0] - _x + 1)

            # 論文執筆用に，ルビ除去後画像とヒストグラムの画像出力もさせている
            # おそくなるので必要なかったらrubi.pyも替えてね
            trim_img = bw_img[_y:int(_y+_h+1), _x:int(_x+_w+1)]

            
            split = rubi.elim_rubi(trim_img, hist_output_dir+str(i)+"-"+str(j)+"_")
            
            if split:
                _w = int(split + 10)
                new_img = trim_img[: ,:split+10]
                # ルビ除去後画像保存 必要なければ消すこと
                cv2.imwrite(rubi_output_dir+str(i)+"-"+str(j)+".jpg", new_img)

            # 左方向，上方向，下方向に拡大する
            hp = img_h//500
            wp = img_w//330
            x, y = int(_x-wp), int(_y-hp)
            if x < 0: x = 0
            if y < 0: y = 0
            h = _h + hp + hp
            w = _w + wp
            
            AddElement(page, str(w), str(h), str(x), str(y)) # 要素追加
        
        tree = gfg.ElementTree(root)
        
        xml_name = output_dir+"xml/"+name+".xml"
        print(xml_name)
        with open (xml_name, "wb") as files :
            tree.write(files)
        
"""
