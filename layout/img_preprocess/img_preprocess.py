import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
    
def threshold(img):

    if len(img.shape)==3:
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
        
    # 大津の二値化
    #ret, img_th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    # ret: 閾値を返す
    # 適応的閾値処理
    img_th = cv2.adaptiveThreshold(img_gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   55, 15)
    return img_th


def molph(bw_img, horizontal, vertical, black, mode):

    if not(black):
        black_img = cv2.bitwise_not(bw_img)
    else:
        black_img = bw_img.copy()

        
    h, w = bw_img.shape[:2]

    if vertical > h: k_ver = 3
    else: k_ver = h//vertical
    if horizontal > w: k_hor = 3
    else:  k_hor = w//horizontal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_hor, k_ver))

    if mode=="dilate":
        bw_molph = cv2.dilate(black_img, kernel, iterations=1)
    elif mode=="erode":
        bw_molph = cv2.erode(black_img, kernel, iterations=1)
    elif mode=="opening":
        bw_molph = cv2.morphologyEx(black_img, cv2.MORPH_OPEN, kernel)
    elif mode=="closing":
        bw_molph = cv2.morphologyEx(black_img, cv2.MORPH_CLOSE, kernel)
        

    if not(black):
        return cv2.bitwise_not(bw_molph)
    else:
        return bw_molph
       

def elim_line(img):

    if len(img.shape)==3:
        img_gray = cv2. cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
        
    bw = cv2.adaptiveThreshold(img_gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,
                               3, 5)
    
    bw = cv2.bitwise_not(bw)

    h, w = bw.shape[:2]

    # 横幅を計算
    horizontal_size = int(w / 70)
    # 縦幅を計算
    vertical_size = int(h / 60)
    #print(horizontal_size, vertical_size)
    
    # カーネルを定義
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    # erode, dilateを実行(Opening)
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontalStructure)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, verticalStructure)

    # 縦線横線を統合
    #elim_line = horizontal+vertical
    close_hor = molph(horizontal, 10, h, black=True, mode="closing")
    dilate_hor = molph(close_hor, w, 200, black=True, mode="dilate")
    dilate_ver = molph(vertical, 700, 100, black=True, mode="dilate")

    img_th = threshold(img)
    src = img_th.copy()
    #src = molph(src, 600, 700, black=False, mode="closing")

    elim_hor = line_contours(src, dilate_hor)
    pre = line_contours(elim_hor, dilate_ver)
    
    #pre = molph(elim_hor_ver, 500, 1000, black=False, mode="closing")
    #pre = molph(elim_hor_ver, 1000, 2000, black=False, mode="erode")
    #pre = molph(pre, 500, h, black=False, mode="closing")
    #plt.imshow(pre, cmap="gray")
    #plt.show()
    return pre


def line_contours(img, bw):
    h, w = img.shape[:2]
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # 輪郭の面積を求める
        area = cv2.contourArea(contour, True)
        #print(f"面積[{i}]: {area}")
    #print((h*0.003)*(w*0.4))
    contours2 = list(filter(lambda x: cv2.contourArea(x) >= 20*(w*0.1), contours))
    img_contours = cv2.drawContours(img, contours2, -1, 255, -1, cv2.LINE_AA)
    #plt.imshow(img_contours, cmap="gray")
    #plt.show()

    return img_contours


"""
def main():    
    #実行プログラム
    
    INPUT_DIR = "./input/Doho_19400201/"
    dirname = os.path.basename(os.path.dirname(INPUT_DIR))
    
    OUT_DIR = "./output/"+dirname+"/"
    os.makedirs(OUT_DIR+"img_pre/", exist_ok=True)
    os.makedirs(OUT_DIR+"npy/", exist_ok=True)

    list_imgdir = [name for name in glob.glob(INPUT_DIR+"*.jpg")]

    for n in list_imgdir:
        # input image
        filename, _ = os.path.splitext(os.path.basename(n))
        img = cv2.imread(n) # 0 -> gray
        #otsu = otsu_threshold(img)
        elim = elim_line(img)
        np.save(OUT_DIR+"npy/"+filename, elim)
        cv2.imwrite(OUT_DIR+"img_pre/"+filename+".jpg", elim)
        
    print("create preprocess file")
        
    
    

if __name__ == "__main__":
    main()
"""
