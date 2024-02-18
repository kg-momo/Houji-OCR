#craft_utilsの出力trimに対応した xmlファイルを作成するプログラム

import os
import glob

import cv2
import numpy as np

import xml.etree.ElementTree as gfg

def XMLGenerator(tree, imagename, width, height):
    # ツリーを取得
    root = tree.getroot()
    
    #要素を追加
    page = gfg.SubElement(root, "PAGE",
                          {"HEIGHT": height,
                           "IMAGENAME": imagename,
                           "WIDTH": width})

    #要素を追加
    line = gfg.SubElement(page, "LINE",
                          {"CONF": "1.000",
                           "HEIGHT": height,
                           "TYPE": "本文",
                           "WIDTH": width,
                           "X": "0",
                           "Y": "0"})
    #ElementTreeを追加
    new_data = root
    
    return new_data

if __name__ == "__main__":

    """
    # 連番でやりたいのでやめた
    img_dir = "./output/trim/test_recognition/"
    img_type = "*.jpg"
    datas = []

    for image_path in glob.glob(os.path.join(img_dir, "img/", img_type)):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # filename
        n = image_path.replace(img_dir, "")
        h, w = image.shape
        # (imagename, width, height)
        data = {"name": n, "width": str(w), "height": str(h)}
        datas.append(data)
    """

    img_dir_pass = "./output/trim/test02/img/"
    file_num = (sum(os.path.isfile(os.path.join(img_dir_pass,name)) for name in os.listdir(img_dir_pass)))
    img_pass = "res_tdo_19400201_0001"
    datas = []
    
    for i in range(file_num):
        # filepass
        file_pass = img_dir_pass+img_pass+"-"+str(i)+".jpg"
        image = cv2.imread(file_pass, cv2.IMREAD_GRAYSCALE)
        # filename
        n = file_pass.replace(img_dir_pass, "")
        h, w = image.shape
        # (imagename, width, height)
        data = {"name": n, "width": str(w), "height": str(h)}
        datas.append(data)

    n = 50
    
    for num in range(0, len(datas), n):
        root = gfg.Element("OCRDATASET")
        tree = gfg.ElementTree(root)
        
        sprit_data = datas[num:num + n]
        for im_d in sprit_data:
            new_data = XMLGenerator(tree,
                                    im_d["name"], im_d["width"], im_d["height"])
            tree = gfg.ElementTree(new_data)

        #ファイルの書き込み
        xml_file_pass = "./output/trim/test02/xml/res_tdo_19400201_0001_"+str(num)+"-"+str(num+len(sprit_data)-1)+".xml"
        
        with open (xml_file_pass, "wb") as files :
            tree.write(files)
