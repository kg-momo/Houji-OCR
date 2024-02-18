import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import io
import glob
from . import edge_fourier as edg
from . import img_preprocess as pre
from PIL import Image
from scipy import signal
from scipy.optimize import curve_fit
import sys

# フーリエ記述子による輪郭線をnumpy配列に格納
def fourier_result(f_list, width, height, img, out_dir):
    #print("h, w: ", height, width)
    fig, ax = plt.subplots(figsize=(width/50, height/50), dpi=240)
    ax.set_xlim(width/(-2),width/2)
    ax.set_ylim(height/(-2), height/(2))
    for f in f_list:
        ax.plot(f.real, f.imag, color="0", lw=1)
        ax.fill(f.real, f.imag, color="0")
        
    plt.axis('off')
    # 確認用
    #plt.savefig(out_dir+"fourier_result.png", bbox_inches='tight', pad_inches=0.0)
    # PIL画像に変換
    buf = io.BytesIO()  
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.0, dpi=240) # 解像度は調整
    plt.close(fig)
    
    buf.seek(0) # バッファの先頭に移動
    pill_img = Image.open(buf).convert('L') # RGBAになるので変換
    numpy_img = cv2.resize(np.array(pill_img), dsize=(width, height))
    # 確認用
    #cv2.imwrite(out_dir+"test.png", numpy_img)

    return numpy_img

# 縦方向濃度ヒストグラムの取得
def Projection_V(img):
    height, width = img.shape[:2]
    array_V = np.zeros(width)
    for i in range(width):
        total_count = 0
        for j in range(height):
            temp_pixVal = img[j, i]
            if (temp_pixVal == 0):
                total_count += 1
        array_V[i] = total_count
 
    return array_V
    
def plot_bar(array, output_dir, rubi):
    width = len(array)

    x = np.arange(width)
    y = np.array([array[i] for i in range(len(array))])
    #median = np.median(y)
    max_darkness = np.amax(y)
    minid = signal.argrelmin(y)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y)
    ax.scatter(x[minid], y[minid], s=15, c="r")
    if not(rubi): pass
    else:
        ax.plot(x[rubi], y[rubi], marker='.', markersize=20, color="r")
    
    ax.axvline(width*(0.6), ls="--", color="navy")
    ax.axhline(max_darkness*(1/2), ls="-.", color="navy")
    plt.title("Result="+str(rubi)+" width="+str(width))
    #plt.show()
    fig.savefig(output_dir+"hist_V.png")
    plt.close()

#########################################
# 濃度ヒストグラムからルビ有無の判定
#########################################
def hantei(array):
    
    width = len(array)
    x = np.arange(width)
    y = np.array([array[i] for i in range(len(array))])

    minid = signal.argrelmin(y)[0] # 濃度ヒストグラムから極小値(id)を取得
    # 一番値(濃度値)の小さい極小値を取得
    if len(minid) >= 1:
        min_minid = minid[0]
        if len(minid) > 1:
            for i in range(1, len(minid)):
                if y[min_minid]>y[minid[i]]:
                    min_minid = minid[i]
        # (ヒストグラムの最大濃度値)*1/2 より小さいか？
        max_darkness = np.amax(y)
        if y[min_minid] > max_darkness*(1/2):
            return False
        # 位置(id)は(画像幅)*0.6より右側にあるか？
        elif min_minid < width*(0.6):
            return False
        else:
            return min_minid
        
    else: return False
    
# フーリエ記述子による輪郭線をnumpy配列に格納
def get_fourier_np(f_list, width, height, img):
    #print("h, w: ", height, width)
    fig, ax = plt.subplots(figsize=(width/50, height/50), dpi=240)
    ax.set_xlim(width/(-2),width/2)
    ax.set_ylim(height/(-2), height/(2))
    for f in f_list:
        ax.plot(f.real, f.imag, color="0", lw=1)
        ax.fill(f.real, f.imag, color="0")
        
    plt.axis('off')
    
    # PIL画像に変換
    buf = io.BytesIO()  
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.0, dpi=240) # 解像度は調整
    plt.close(fig)   
    buf.seek(0) # バッファの先頭に移動
    pill_img = Image.open(buf).convert('L') # RGBAになるので変換
    numpy_img = cv2.resize(np.array(pill_img), dsize=(width, height))
    
    return numpy_img

# モジュールとして使うとき用 
def get_rubi_hantei(img):
    
    height, width = img.shape[:2]

    list_contours = pre.preprocess(img)

    #####################
    
    f_list = []
    for cnt in list_contours:
        # フーリエ記述子による輪郭線取得
        f = edg.get_fourier_transform(cnt)
        f_list.append(f)
    
    # fourier 輪郭線 画像(ndarray)
    np_img = get_fourier_np(f_list, width, height, img)
    prj_v = Projection_V(np_img)

    rubi = hantei(prj_v)
    
    #x = np.arange(width)
    #y = np.array([prj_v[i] for i in range(len(prj_v))])
    
    if not(rubi): # rubi False
        return False
    else:
        return rubi
    
    
    
def main():    
    #実行プログラム
    INPUT_DIR = sys.argv[1]
    DIR = os.path.basename(os.path.dirname(INPUT_DIR))

    OUT_DIR = "./output/"+DIR+"/hantei_hist/"
    os.makedirs(OUT_DIR, exist_ok=True)

    list_imgpath = [name for name in glob.glob(INPUT_DIR+"*.jpg")]

    for i, name in enumerate(list_imgpath):    
        filename, _ = os.path.splitext(os.path.basename(name))

        # input image
        img = cv2.imread(name)
        height, width = img.shape[:2]
        
        # convert gray scale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 比較用
        # black white(OTSUthreshold)
        ret, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
        #bw_img = 255-bw_img # 白黒反転
        
        ### 画像の前処理 ###
        list_contours = pre.preprocess(img)
        
        f_list = []
        #### フーリエ記述子による輪郭線取得 ###
        for cnt in list_contours:
            f = edg.get_fourier_transform(cnt)
            f_list.append(f)
            
        print("{:d}/{:d}: {:s}\r".format(i+1, len(list_imgpath), name), end='')
        # フーリエ記述子輪郭線ndarray変換
        np_img = get_fourier_np(f_list, width, height, img)
        prj_v = Projection_V(np_img)

        rubi = hantei(prj_v)
        plot_bar(prj_v, OUT_DIR+filename+"_", rubi)
        
        result_line = cv2.line(img, pt1=(rubi, 0), pt2=(rubi, height), color=(0,0,255), thickness=1)
        # ルビの判定箇所を可視化
        cv2.imwrite(OUT_DIR+filename+"_line.jpg", result_line)
        # フーリエ記述子による輪郭（中うめ後）可視化
        cv2.imwrite(OUT_DIR+filename+"_fourier.jpg", np_img)

    
if __name__ == "__main__":
    main()
