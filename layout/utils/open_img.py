from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
 
import numpy as np
import sys
import os
import glob
import cv2
import shutil

# craft_utilsの結果を
# 画像として出力するためのプログラム


# 矩形を長方形のみの想定に変更している
# boxの座標は、左上開始、右回り4点の座標

# 行切り出しか、セクション推定かで
# SPLIT_SEG を変更すること！

canvas_size = 5000

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

    return resized, ratio, size_heatmap

def img_quadbox(img, boxes, name, outf, SPLIT_SEG):
    img_box = np.copy(img)
    # QuadBoxを重ねる
    for i in range(len(boxes)):
        plot = boxes[i].astype('uint64')
        cv2.polylines(img_box, [plot], True, (0, 30, 230), 2)

    if SPLIT_SEG:  
        output_dir = "./output/box_seg/"+outf # box or box_seg
    else:
        output_dir = "./output/box/"+outf # box or box_seg

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
        
    cv2.imwrite(output_dir+name+"_box.jpg", img_box)


def trim_quadbox(img, boxes, name, outf):
    # QuadBoxを利用して行切り出し
    # img[top : bottom, left : right]でトリミング
    # img[y:y+h, x:x+w]でもできる
    # boxesの各要素(plot)-> [0]:左上座標 [1]:右上 [2]:右下 [3]:左下
    # opencvの座標って左上から始まってるpoi

    output_dir = "./output/trim/"+outf
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    
    ratio_plot = []
    
    for i in range(len(boxes)):
        plot = boxes[i].astype('uint64')
        x, y = plot[0][0], plot[0][1]
        h = int(plot[3][1] - y -1)
        w = int(plot[1][0] - x -1)

        ratio = abs(w/h)
        ratio_plot.append(ratio)

        if ratio < 0.4:
            img_trim = img[y:int(y+h+1), x:int(x+w+1)]
            cv2.imwrite(output_dir+name+"_"+str(i)+".jpg", img_trim)

        ratio = abs(w/h)
        ratio_plot.append(ratio)
        
        #cv2.namedWindow("img_trim", cv2.WINDOW_NORMAL)
        #cv2.imshow('img_trim', img_trim)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    plt.plot(ratio_plot, ".")
    plt.show()
            
if __name__ == "__main__":


    SPLIT_SEG = False

    if SPLIT_SEG:
        FILE_NAME = "./output/data_seg/elim_line/" # data or data_seg
    else:
        FILE_NAME = "./output/data/elim_line/" # data or data_seg
        
    OUT_F = "elim_line/"

    b_path = []
    p_path = []
    m_path = []
    l_path = []
    name_list = []
    
    for name in glob.glob(FILE_NAME+"*_boxes.npy"):
        b_path.append(name)
        p_path.append(name.replace("boxes", "polys"))
        m_path.append(name.replace("boxes", "mapper"))
        l_path.append(name.replace("boxes", "labels"))
        _filename, _ = os.path.splitext(os.path.basename(name))
        filename = _filename.replace("-np_boxes", "")
        name_list.append(filename)

    for i, name in enumerate(name_list):
        # 必要データの読み込み
        boxes = np.load(b_path[i])
        polys = np.load(p_path[i], allow_pickle=True)
        mapper = np.load(m_path[i])
        labels = np.load(l_path[i])
        
        print(f"{i} / {len(name_list)}: {name}")
        
        # 重ねる画像
        match = name.replace("res_", "")
        img_path = glob.glob("./HojiShinbun/Doho_19400201/"+match+"*")
        print("img:", img_path)
        img = cv2.imread(img_path[0])
        
        # 画像のリサイズ
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        cv2.imwrite("./output/resize/"+name+"_resize.jpg", img_resized)
        
        # 白地のキャンバスを用意 # 使ってない
        #canvas = np.ones((labels.shape), np.uint8) * 255
        #canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGBA)
        
        # QuadBoxの可視化
        #img_quadbox(img_resized, boxes, name, OUT_F, SPLIT_SEG)
        
        # ラベリング処理後の出力可視化
        #cv2.imwrite("./output/label/"+name+"_label.jpg", labels)
        
        # QuadBoxを利用して行切り出し
        trim_quadbox(img_resized, boxes, name, OUT_F)
