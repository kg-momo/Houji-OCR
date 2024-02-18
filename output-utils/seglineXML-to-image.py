# xml-to-imageとほぼいっしょ
# セクション(box)ごと分けられたdata_segline-xmlを
# 可視化するためのprogram

# 2023/10/12 テキストファイル出力機能追加

import cv2
import numpy as np
import sys
import os
import xml.etree.ElementTree as gfg
from PIL import Image, ImageDraw, ImageFont
import glob
import re
import shutil
import jsons
from argparse import ArgumentParser
import seg_dict
import make_segline_xml as make_xml

def get_options(args):
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str,
                        help='Resize image Directory')
    parser.add_argument('--out_dir', type=str,
                        help='Output dir')
    options = parser.parse_args(args)
    return options

# xmlファイルからファイル名(img)を取り出す
def xml_getName(root, f_list, f_name):

    for pagename in root.iter("PAGE"):
        name = pagename.attrib["IMAGENAME"]
        f_list.append(name)
        _filename, _ = os.path.splitext(os.path.basename(name))
        f_name.append(_filename)
        
# xmlファイルからデータboxes{x, y, w, h, name}を取り出す
def xml_getBox(root, boxes):

    for box in root.iter("BOX"):

        b_name = box.attrib["BOXNAME"]
        b_x = box.attrib["X"]
        b_y = box.attrib["Y"]
        b_w = box.attrib["WIDTH"]
        b_h = box.attrib["HEIGHT"]

        b_data = {"name": b_name, "height": int(b_h), "width": int(b_w),
                  "x": int(b_x), "y": int(b_y)}

        boxes.append(b_data)
    
# xmlファイルから特定のboxのデータdatas{x, y, w, h, name}を取り出す
def xml_getData(root, box_name, datas):

    for box in root.iter("BOX"):
        b_name = box.attrib["BOXNAME"]
        if b_name == box_name:
            for line in box.iter("LINE"):
                s = line.attrib["STRING"]
                h = line.attrib["HEIGHT"]
                w = line.attrib["WIDTH"]
                x = line.attrib["X"]
                y = line.attrib["Y"]                
                data = {"string": s, "height": int(h), "width": int(w),
                        "x": int(x), "y": int(y)}                
                datas.append(data)

           
def output_text(boxes, img_pth, out_dir, name):

    im = Image.open(img_pth).convert(mode="RGB")
    font_name = "fontsfile/ipamp.ttf"
    box_font = ImageFont.truetype(font_name, 15)
    font = ImageFont.truetype(font_name, 15)
    im_w, im_h = im.size
    d = ImageDraw.Draw(im)

    for box in boxes:        
        n, h, w, x, y, lines = box.values()
        color = tuple(np.random.choice(range(256), size=3))
        boxname_position = (x + w, y)
        d.text(boxname_position, n, color, font=box_font)

        draw_box(x, y, h, w, im, color)
        
        draw_text(lines, im, color, font)
        

    im.save(out_dir+name+"-boxText.jpg")


def draw_box(x, y, h, w, img, color):
    d = ImageDraw.Draw(img)
    d.rectangle(((x, y), (x+w-1, y+h-1)), fill=None, outline=color, width=10)

def draw_text(lines, img, color, font):    
    d = ImageDraw.Draw(img)
    for line in lines:        
        #print(line.keys())
        # -->> dict_keys(['string', 'height', 'width', 'x', 'y'])
        text = line["string"]
        tx = line["x"]+line["width"]-2
        ty = line["y"]
        d.text((tx, ty), text, (0, 0, 110), font=font, direction="ttb")

        
def output_textfile(boxes, out_dir, name):

    out_txtfile = out_dir+name+"_ordertext.txt"
            
    with open(out_txtfile, mode='w', encoding='utf-8') as f:
        for box in boxes:
            _, _, _, _, _, lines = box.values()
            f.write("\n")
            for line in lines:
                text = line["string"]
                f.write(text+"\n")
        
if __name__ == "__main__" :
    
    try:
        options = get_options(sys.argv[1:])
    except FileNotFoundError as err:
        print(err)
        sys.exit()

    DIR_PATH = options.dir_name #->output/$name/
    OUT_PATH = options.out_dir
    out_dir = OUT_PATH+"result/seg/"
    os.makedirs(out_dir, exist_ok=True)

    text_xml = glob.glob(DIR_PATH+"text/*.xml")[0]
    #print(text_xml)
    
    resize_list = [name for name in glob.glob(DIR_PATH+"imgxml/img/*.jpg")]

    seg_box_list = []
    filename_list = []
    for i, img in enumerate(resize_list):
        filename, _ = os.path.splitext(os.path.basename(img))
        seg_box = glob.glob(DIR_PATH+"seg/boxes/"+filename+"*.npy")[0]
        filename_list.append(filename)
        seg_box_list.append(seg_box)

    segLineXML = []
    for i, name in enumerate(filename_list):
        dic = seg_dict.get_segdict(text_xml, seg_box_list[i], name+".jpg")
        segLineXML.append(make_xml.segline_xml(resize_list[i], dic, DIR_PATH+"seg/"))
        
    for i, name in enumerate(filename_list):        
         # 重ねる画像（リサイズ済み）
        img_path = resize_list[i]
        #print(img_path)
        # XMLfile
        xml_tree = gfg.parse(segLineXML[i])
        # get root
        root = xml_tree.getroot()

        # GetSegbox
        boxes = []       
        xml_getBox(root, boxes)

        for j, box in enumerate(boxes):
            # data
            datas = []
            xml_getData(root, box["name"], datas)
            boxes[j]["LINES"] = datas

        #print(boxes[0].keys())
        # -->> dict_keys(['name', 'height', 'width', 'x', 'y', 'LINES'])

        output_text(boxes, img_path, out_dir, name)
        output_textfile(boxes, out_dir, name)
