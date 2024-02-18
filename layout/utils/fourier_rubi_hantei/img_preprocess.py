import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
import statistics
import os
import shutil
import sys
import glob

def align_points(cnt00):
    
    # 輪郭に沿った長さの列 
    lengs = [cv2.arcLength(cnt00[:i+1], False)  for i in range(len(cnt00))]
    
    # 輪郭点を分割
    SPANS = 128
    allLength = cv2.arcLength(cnt00,True)
    needLengs = np.linspace(0,allLength,SPANS) # 分割した場合の弧距離のリスト
    lengs.append(allLength)
    cnt00 = np.r_[cnt00,[cnt00[0]]]
    s_indexies = []
    index = 0
    for i in range(SPANS):
        nl = needLengs[i]
        for j in range(index,len(cnt00)-1):
            l0,l1 = lengs[j],lengs[j+1]
            if l0 <= nl and nl <= l1:
                if np.sqrt((l0-nl)**2) < np.sqrt((l1-nl)**2):
                    s_indexies.append(j)
                    index = j+1
                else:
                    s_indexies.append(j+1)
                    index = j+2
                break
    samples =  np.array([[cnt00[i][0][0],cnt00[i][0][1]]  for i  in s_indexies])
    
    
    # 表示して確認
    #plt.figure(figsize=(6,6),dpi=100)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.gca().invert_yaxis() 
    #plt.scatter(samples[:,0],samples[:,1] ,marker='.',color="red")
    #plt.show()

    return samples

        

# 輪郭点の取得
def get_contours(img):
    
    height, width = img.shape[:2]
    # convert gray scale image
    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.uint8)

    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

    
    img_binary = 255-img_binary # 白黒反転

    # ルビを消してしまう可能性があったので使っていません
    # ノイズ除去処理
    #ksize = 3
    #img_mask = cv2.medianBlur(img_binary, ksize) # 中央値フィルタ
    
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(x) >= 5, contours))
    ### contours[輪郭番号][点の番号][0][X座標, Y座標] ###

    # 座標の点数多すぎるもの(128以上)は減らす
    p128_contours = [align_points(contours[i]) for i in range(len(contours))]

    return p128_contours

def preprocess(img):

    contours = get_contours(img)
    height, width = img.shape[:2]
    
    ######
    # 輪郭ごと出力するlist
    list_contours = []
    
    for i, cnt in enumerate(contours):

        point_x = np.array([r[0] for r in cnt])
        point_y = np.array([r[1]*(-1) for r in cnt])
        
        #画像の中心を原点にそろえる
        x = point_x - width/2
        y = point_y + height/2
        
        xy_contour = np.array([x[num] + y[num]*1j for num in range(len(x))])

        list_contours.append(xy_contour)

    #print("contours num: ", len(list_contours))
        
        
    return list_contours


def main():

    INPUT_DIR = sys.argv[1]
    DIR = os.path.basename(os.path.dirname(INPUT_DIR))
    
    OUT_DIR = "./output/"+DIR+"/img_preprocess/"
    os.makedirs(OUT_DIR, exist_ok=True)

    
    list_imgpath = [name for name in glob.glob(INPUT_DIR+"*.jpg")]

    for name in list_imgpath:
        # input image
        img = cv2.imread(name)
        height, width = img.shape[:2]
    
        contours = get_contours(img)
        
        filename, _ = os.path.splitext(os.path.basename(name))
        dst = np.ones((height, width, 3), np.uint8) * 255
        img_contours = cv2.drawContours(dst, contours, -1, (255,0,0), 1)
        cv2.imwrite(OUT_DIR+filename+"_contours.jpg", img_contours)
    


if __name__ == "__main__":
    main()
