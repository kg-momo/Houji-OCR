from numpy.matrixlib.defmatrix import N
from sklearn.preprocessing import StandardScaler
from matplotlib import image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from functools import partial
from scipy import interpolate
import os
import shutil
from . import img_preprocess as pre
import glob
import sys

#高速フーリエ変換関数（時間間引き形のfftを採用）
def fft(x_n):  
  N = len(x_n)
  print(N)
  n = N//2

  #if N == 1:
  if N <= 1:
    return x_n[0]

  f_even = x_n[0:N:2]  
  f_odd = x_n[1:N:2]   
  F_even = np.array(fft(f_even))
  F_odd = np.array(fft(f_odd))  

  #0<=k<=N/2-1のWを計算
  W_N =  np.exp(-1j * (2 * np.pi * np.arange(0,n)) / N)   

  X_k = np.zeros(N, dtype ='complex')

  print(F_even +  F_odd * W_N)

  X_k[0:n] = F_even +  F_odd * W_N  
  X_k[n:N] = F_even -  F_odd * W_N    

  return X_k


#フーリエ級数展開を行う関数 G型フーリエ記述子
def fft_integral(X_k,N,x_n):
  center_X = len(X_k) // 2
  X_k = X_k[center_X - N : center_X + N+1]

  ts = np.linspace(
        0.0, 2.0 * np.pi, len(x_n)
  ) - np.pi
  f = []
  
  for t in ts:
    temp = np.array(
      [X_k[i] * np.exp(1j * k * t) for i, k in enumerate(range(-N, N+1))]
    )
    f.append(temp.sum())
  f = np.array(f)
  return f

#  結果の表示
def plot_f(contour, img, f, output):

  point_X = contour.real
  point_Y = contour.imag
  height, width = img.shape[:2]

  fig, ax = plt.subplots(1, 2, tight_layout=True)
  #ax = plt.figure(num=0, dpi=240, figsize=(height/50, width/50)).gca()
  #ax[0] = plt.figure(num=0, dpi=240).gca()
  ax[0].set_xlim(width/(-2),width/2)
  ax[0].set_ylim(height/(-2), height/(2))
  
  ax[0].plot(point_X, point_Y, lw=1)
  ax[0].scatter(point_X, point_Y, s=1, color="red") #, label=file_name)
  ax[0].set_aspect('equal', adjustable='box')

  ax[0].imshow(img, extent=[*ax[0].get_xlim(), *ax[0].get_ylim()], alpha=0.6)
  
  #ax = plt.figure(num=1, dpi=240, figsize=(5, 7)).gca()
  ax[1].scatter(f.real, f.imag, c="red")
  ax[1].plot(f.real, f.imag, lw=1, label='N='+str(len(f)))
  ax[1].grid()
  ax[1].legend(loc=0)


  plt.savefig(output+"_countour_and_fourier.png")
  #plt.show();
  plt.close()


#  結果の表示 全体のフーリエ級数展開結果
def plot_result(f_list, width, height, out_dir):
  ax = plt.figure(num=None, dpi=240, figsize=(width/50, height/50)).gca()
  ax.set_xlim(width/(-2),width/2)
  ax.set_ylim(height/(-2), height/(2))
  for f in f_list:
    ax.plot(f.real, f.imag, lw=1)
  ax.grid()
  plt.savefig(out_dir+"fourier_result.png", bbox_inches='tight')
  plt.close()


# フーリエ記述子の取得
def get_fourier_transform(contours):

  x_n = contours.copy()
  
  #高速フーリエ変換を行う
  x_uv = np.fft.fft(x_n)
  X_k = np.fft.fftshift(x_uv) / len(x_n)

  #次数の決定
  num=(int(len(X_k) // 2)-1) # 最大次数
  # 指定する場合は以下を有効にする
  if 10 < num:
    num = 10

  f = fft_integral(X_k, num, x_n)
  # x: 実部(f.real) y: 虚部(f.imag)
  return f
  
def main():    
    
  #実行プログラム
  INPUT_DIR = sys.argv[1]
  DIR = os.path.basename(os.path.dirname(INPUT_DIR))
  
  OUT_DIR = "./output/"+DIR+"/edge_fourier/"
  os.makedirs(OUT_DIR, exist_ok=True)

  list_imgpath = [name for name in glob.glob(INPUT_DIR+"*.jpg")]
  
  for name in list_imgpath:
    # input image
    img = cv2.imread(name)
    height, width = img.shape[:2]
    
    filename, _ = os.path.splitext(os.path.basename(name))
    os.makedirs(OUT_DIR+filename, exist_ok=True)
    ### 前処理 ###
    # 輪郭点listの取得
    list_contours = pre.preprocess(img)

    f_list = []
    for i, contours in enumerate(list_contours):
      print("{:d}/{:d}: {:s}".format(i+1, len(list_contours), name), end='\r')
      f = get_fourier_transform(contours)
      plot_f(contours, img, f, OUT_DIR+filename+"/"+str(i).zfill(3))
      f_list.append(f)
      
    plot_result(f_list, width, height, OUT_DIR+filename+"/")
    
  
if __name__ == '__main__':
  main()
